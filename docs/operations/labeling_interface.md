# KML 打标接口与批处理交付

## 交付形态

评审方在 KML 容器内运行项目源码。模型权重与代码分离，推理只访问容器内模型服务，
不调用第三方 API。推荐使用批处理 CLI；HTTP `/api/infer` 用于单条联调。

## 输入 Schema

最小字段为 `item_id` 与 `query`。程序兼容公开300条数据的完整字段，包括：

- `doc_title`、`doc_caption`、`doc_author`、`doc_kwaiid`
- `doc_asr`、`doc_ocr`、`doc_ocr_all`
- `doc_hashtag`、`doc_mllm_queries`、`doc_mllm_query1`～`doc_mllm_query5`
- `doc_hetu_tag_id1`、`doc_text_source`、`create_time`
- `blobstore_key`、`url_path`、`video_url`

Gold、query含义、人工视频内容说明不得作为推理输入。隐藏集视频由评审方准备；程序按
`<video-dir>/<item_id>.mp4`（也支持 mov/mkv/webm）匹配，不自行访问外部信息源。

## 批量命令

```bash
cd /code
export PYTHONPATH=/code/src
python scripts/run_evaluation_batch.py \
  --input /data/model_input.json \
  --video-dir /data/videos \
  --output /results/predictions.jsonl \
  --frame-root /results/frames \
  --endpoint http://127.0.0.1:8001/v1/chat/completions \
  --model Qwen3-VL-32B-Instruct
```

输入支持 JSON、JSONL、CSV、XLSX；输出为可断点续跑的 JSONL。重复执行同一命令时，
已成功的 `item_id` 会被跳过，失败样本保留错误类型与消息。
默认禁止从 `video_url` 下载，避免隐藏数据离开获批环境；如评审方明确提供允许下载的内部
直链，可显式增加 `--allow-video-download`。使用 `--require-video` 可在缺少本地视频时直接
阻断，避免把文本降级误报成多模态结果。

## 单条 HTTP 接口

```bash
MODEL_ENDPOINT=http://127.0.0.1:8001/v1/chat/completions \
MODEL_NAME=Qwen3-VL-32B-Instruct \
bash scripts/start_web_demo.sh

curl -X POST http://127.0.0.1:7860/api/label \
  -H 'Content-Type: application/json' \
  -d '{"item_id":"150101226827701","query":"brisa otto conhece primeiro vez","video_path":"/data/videos/150101226827701.mp4"}'
```

输出至少包含 `sample_id`、`relevance_label`、`confidence`、`reason`，并附带证据、分层、
模型版本、Prompt版本、延迟、帧数和处理状态。

## 性能与 QPS

当前 Qwen3-VL-32B Transformers 参考部署使用单请求生成锁，8帧/条，实测约45秒/条，
稳定吞吐约0.02 QPS。不要通过提高HTTP并发绕过模型锁；这只会排队并增加超时。
若评审要求更高吞吐，应在保持同一Prompt/帧配置的前提下，使用经过Vision验证的批处理
推理框架或多个独立模型副本，并重新报告QWK、吞吐和显存。

## 数据与失败策略

- 原始 Gold 永不进入 Prompt。
- 视频缺失时不得宣称完成多模态推理；由调用方决定补齐视频或接受文本降级。
- 非JSON模型输出会进行一次本地格式修复；不会改变原判断或添加事实。
- 模型下载、远程API和训练默认关闭。
