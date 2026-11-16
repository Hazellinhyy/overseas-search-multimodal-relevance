from overseas_search_relevance.cli.main import build_parser


def test_cli_groups_exist() -> None:
    parser = build_parser()
    assert parser.parse_args(["report"]).group == "report"
    assert parser.parse_args(["infer"]).group == "infer"
