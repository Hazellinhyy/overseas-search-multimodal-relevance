# 看后搜相关性标准

版本：1.0.0

先判断 Query 与 Refer Video 的 strong、weak、unrelated 关系，再独立判断 Query-Doc。

- strong/unrelated：`final_score = query_doc_score`
- weak：`final_score = max(query_doc_score, refer_doc_score - 1)`

关系无法判断、前序视频缺失或证据冲突时标为 review，不生成最终分数。

