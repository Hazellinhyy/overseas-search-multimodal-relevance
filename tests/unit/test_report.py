from overseas_search_relevance.evaluation.report import build_report


def test_report_uses_real_metrics_and_cases():
    rows = [
        {
            "sample_id": "1",
            "processing_status": "success",
            "gold_label": 3,
            "relevance_label": 2,
            "confidence": 0.8,
            "tier": "B",
            "latency_ms": 100,
            "reason": "missing qualifier",
        },
        {
            "sample_id": "2",
            "processing_status": "success",
            "gold_label": 0,
            "relevance_label": 0,
            "confidence": 0.9,
            "tier": "A",
            "latency_ms": 200,
        },
    ]
    report = build_report(rows)
    assert "±1 Adjacent Accuracy" in report
    assert "Case 1" in report
    assert "待复核" in report


def test_report_does_not_invent_missing_gold():
    report = build_report([])
    assert "不伪造结果" in report


def test_report_excludes_discarded_gold_and_reports_missing_media():
    report = build_report(
        [
            {
                "sample_id": "discarded",
                "processing_status": "success",
                "gold_status": "discarded",
                "media_status": "missing",
                "relevance_label": 1,
                "tier": "C",
                "latency_ms": 1,
            }
        ]
    )
    assert "有效 Gold：0" in report
    assert "官方丢弃：1" in report
    assert "视频缺失/文本降级：1" in report
