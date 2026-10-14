from unipath.fees import load_fee_table, match_universities
from unipath.tools import compare_fees


def compare(universities: list[str]) -> str:
    return compare_fees.invoke({"universities": universities})


def test_fee_table_is_parsed_from_markdown():
    rows = load_fee_table()
    assert len({r.university for r in rows}) == 5
    eastmere_ify = next(
        r for r in rows
        if r.university == "University of Eastmere" and r.programme == "International Foundation Year"
    )
    assert (eastmere_ify.tuition_gbp, eastmere_ify.deposit_gbp) == (14950, 3000)
    assert eastmere_ify.source == "eastmere_university.md"


def test_compares_selected_universities_and_cites_sources():
    result = compare(["Kingsford", "Riverton"])
    assert "£21,500" in result and "£13,500" in result
    assert "[source: kingsford_university.md]" in result
    assert "[source: riverton_university.md]" in result
    assert "Northbridge" not in result
    assert "Cheapest International Foundation Year: Riverton University (£13,500)" in result


def test_empty_list_compares_all_universities():
    result = compare([])
    for name in ["Northbridge", "Kingsford", "Eastmere", "Highmoor", "Riverton"]:
        assert name in result


def test_unknown_university_is_reported_not_guessed():
    result = compare(["Oxford"])
    assert "NO_RELEVANT_RESULTS for: Oxford" in result
    assert "|" not in result  # no fee table invented


def test_name_matching_is_forgiving():
    known = ["University of Eastmere", "Kingsford University London", "Highmoor University"]
    matched, unknown = match_universities(
        ["eastmere uni", "Kingsford University London", "Highmor", "Cambridge"], known
    )
    assert matched == ["University of Eastmere", "Kingsford University London", "Highmoor University"]
    assert unknown == ["Cambridge"]


def test_custom_data_folder(tmp_path):
    (tmp_path / "demo_university.md").write_text(
        "# Demo University — Pathways\n\n## Fees (sample data)\n\n"
        "| Programme | Tuition fee (GBP) | Deposit (GBP) |\n|---|---|---|\n"
        "| Foundation | 9,999 | 1,000 |\n\n## Deadlines\n\n- 1 May\n"
    )
    rows = load_fee_table(tmp_path)
    assert len(rows) == 1
    assert (rows[0].university, rows[0].tuition_gbp) == ("Demo University", 9999)
