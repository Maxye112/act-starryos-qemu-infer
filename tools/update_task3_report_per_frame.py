#!/usr/bin/env python3
"""Insert per-frame QEMU left/right wheel comparison into Proj57任务三报告.docx."""

from __future__ import annotations

import re
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

REPORT = Path(__file__).resolve().parents[1] / "Proj57任务三报告.docx"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
W = f"{{{W_NS}}}"
W14 = f"{{{W14_NS}}}"

ET.register_namespace("w", W_NS)
ET.register_namespace("w14", W14_NS)

FRAMES = [37, 59, 228, 229, 313, 331, 392, 463, 542, 586]

GT = {
    37: (-0.100000, 0.100000, "left"),
    59: (-0.100000, 0.100000, "left"),
    228: (0.100000, -0.100000, "right"),
    229: (0.100000, -0.100000, "right"),
    313: (0.100000, -0.100000, "right"),
    331: (0.100000, -0.100000, "right"),
    392: (0.100000, -0.100000, "right"),
    463: (-0.100000, 0.100000, "left"),
    542: (-0.100000, 0.100000, "left"),
    586: (-0.100000, 0.100000, "left"),
}

INT8_FP16 = {
    37: (-0.115220, 0.109597),
    59: (-0.109223, 0.106129),
    228: (0.119421, -0.115526),
    229: (0.108897, -0.110603),
    313: (0.115606, -0.113134),
    331: (0.116630, -0.115986),
    392: (0.117613, -0.115253),
    463: (-0.106989, 0.105387),
    542: (-0.109660, 0.106409),
    586: (-0.106989, 0.105387),
}

FP32 = {
    37: (-0.114567, 0.109682),
    59: (-0.107739, 0.106333),
    228: (0.119083, -0.114818),
    229: (0.104272, -0.107661),
    313: (0.113086, -0.113270),
    331: (0.115757, -0.117282),
    392: (0.116557, -0.115510),
    463: (-0.110602, 0.106166),
    542: (-0.110222, 0.107406),
    586: (-0.110602, 0.106166),
}


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(f"{W}t"))


def _set_paragraph_text(p: ET.Element, text: str) -> None:
    runs = p.findall(f"{W}r")
    if not runs:
        r = ET.SubElement(p, f"{W}r")
        t = ET.SubElement(r, f"{W}t")
        t.text = text
        return
    ts = runs[0].find(f"{W}t")
    if ts is None:
        ts = ET.SubElement(runs[0], f"{W}t")
    ts.text = text
    for extra in runs[0].findall(f"{W}t")[1:]:
        extra.text = ""
    for run in runs[1:]:
        for t in run.findall(f"{W}t"):
            t.text = ""


def _set_cell_text(cell: ET.Element, text: str) -> None:
    ts = list(cell.iter(f"{W}t"))
    if not ts:
        p = cell.find(f"{W}p")
        if p is None:
            p = ET.SubElement(cell, f"{W}p")
        r = ET.SubElement(p, f"{W}r")
        t = ET.SubElement(r, f"{W}t")
        t.text = text
        return
    ts[0].text = text
    for t in ts[1:]:
        t.text = ""


def _strip_w14_ids(elem: ET.Element) -> None:
    for el in elem.iter():
        for key in list(el.attrib):
            if key == W14 + "paraId" or key == W14 + "textId":
                del el.attrib[key]


def _clone_paragraph(proto: ET.Element, text: str) -> ET.Element:
    p = deepcopy(proto)
    _strip_w14_ids(p)
    _set_paragraph_text(p, text)
    return p


def _serialize_fragment(elements: list[ET.Element]) -> str:
    parts: list[str] = []
    for el in elements:
        parts.append(ET.tostring(el, encoding="unicode"))
    return "".join(parts)


TABLE_TOP_BOTTOM_SZ = "12"
TABLE_HEADER_LINE_SZ = "8"
CELL_MARGIN_TWIPS = "100"
CELL_LINE = "320"


def _ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _make_border(tag: str, *, val: str = "single", sz: str = TABLE_TOP_BOTTOM_SZ) -> ET.Element:
    border = ET.Element(f"{W}{tag}")
    border.set(f"{W}val", val)
    if val != "nil":
        border.set(f"{W}sz", sz)
        border.set(f"{W}space", "0")
        border.set(f"{W}color", "000000")
    return border


def _clear_run_bold(run: ET.Element) -> None:
    rpr = run.find(f"{W}rPr")
    if rpr is None:
        return
    bold = rpr.find(f"{W}b")
    if bold is not None:
        rpr.remove(bold)


def _set_cell_paragraph_layout(p: ET.Element, *, header: bool) -> None:
    ppr = _ensure_child(p, f"{W}pPr")
    spacing = _ensure_child(ppr, f"{W}spacing")
    spacing.set(f"{W}after", "0")
    spacing.set(f"{W}before", "0")
    spacing.set(f"{W}line", CELL_LINE)
    spacing.set(f"{W}lineRule", "auto")
    jc = _ensure_child(ppr, f"{W}jc")
    jc.set(f"{W}val", "center")
    for run in p.findall(f"{W}r"):
        if header:
            rpr = _ensure_child(run, f"{W}rPr")
            if rpr.find(f"{W}b") is None:
                ET.SubElement(rpr, f"{W}b")
        else:
            _clear_run_bold(run)
        rpr = run.find(f"{W}rPr")
        if rpr is not None:
            sz_cs = rpr.find(f"{W}szCs")
            if sz_cs is not None:
                rpr.remove(sz_cs)


def _apply_cell_margins(tc_pr: ET.Element) -> None:
    for child in list(tc_pr):
        if child.tag == f"{W}tcMar":
            tc_pr.remove(child)
    tc_mar = ET.SubElement(tc_pr, f"{W}tcMar")
    for side in ("top", "bottom"):
        mar = ET.SubElement(tc_mar, f"{W}{side}")
        mar.set(f"{W}w", CELL_MARGIN_TWIPS)
        mar.set(f"{W}type", "dxa")
    for side in ("left", "right"):
        mar = ET.SubElement(tc_mar, f"{W}{side}")
        mar.set(f"{W}w", "60")
        mar.set(f"{W}type", "dxa")


def _apply_three_line_table(tbl: ET.Element) -> None:
    tbl_pr = _ensure_child(tbl, f"{W}tblPr")
    for child in list(tbl_pr):
        if child.tag == f"{W}tblBorders":
            tbl_pr.remove(child)
    tbl_w = tbl_pr.find(f"{W}tblW")
    if tbl_w is not None:
        tbl_w.set(f"{W}w", "0")
        tbl_w.set(f"{W}type", "auto")

    rows = tbl.findall(f"{W}tr")
    if not rows:
        return
    last_idx = len(rows) - 1

    for row_idx, tr in enumerate(rows):
        is_header = row_idx == 0
        is_last = row_idx == last_idx

        for tc in tr.findall(f"{W}tc"):
            tc_pr = _ensure_child(tc, f"{W}tcPr")
            for child in list(tc_pr):
                if child.tag == f"{W}tcBorders":
                    tc_pr.remove(child)
            _apply_cell_margins(tc_pr)
            v_align = _ensure_child(tc_pr, f"{W}vAlign")
            v_align.set(f"{W}val", "center")

            tc_borders = ET.SubElement(tc_pr, f"{W}tcBorders")
            for tag in ("left", "right", "insideH", "insideV"):
                tc_borders.append(_make_border(tag, val="nil"))

            if is_header:
                tc_borders.append(_make_border("top", sz=TABLE_TOP_BOTTOM_SZ))
            else:
                tc_borders.append(_make_border("top", val="nil"))

            if is_header and not is_last:
                tc_borders.append(_make_border("bottom", sz=TABLE_HEADER_LINE_SZ))
            elif is_last:
                tc_borders.append(_make_border("bottom", sz=TABLE_TOP_BOTTOM_SZ))
            else:
                tc_borders.append(_make_border("bottom", val="nil"))

            for p in tc.findall(f"{W}p"):
                _set_cell_paragraph_layout(p, header=is_header)


def _rebalance_table_grid(tbl: ET.Element, ncols: int) -> None:
    grid = tbl.find(f"{W}tblGrid")
    if grid is None:
        grid = ET.Element(f"{W}tblGrid")
        tbl_pr = tbl.find(f"{W}tblPr")
        insert_at = list(tbl).index(tbl_pr) + 1 if tbl_pr is not None else 0
        tbl.insert(insert_at, grid)
    for child in list(grid):
        grid.remove(child)
    col_w = str(max(800, 9000 // ncols))
    for _ in range(ncols):
        ET.SubElement(grid, f"{W}gridCol", {f"{W}w": col_w})
    for tr in tbl.findall(f"{W}tr"):
        for tc in tr.findall(f"{W}tc"):
            tc_pr = tc.find(f"{W}tcPr")
            if tc_pr is None:
                continue
            tc_w = tc_pr.find(f"{W}tcW")
            if tc_w is None:
                tc_w = ET.SubElement(tc_pr, f"{W}tcW")
            tc_w.set(f"{W}w", col_w)
            tc_w.set(f"{W}type", "dxa")


def _find_table_after_caption(body: ET.Element, caption_prefix: str) -> ET.Element | None:
    children = list(body)
    for idx, child in enumerate(children):
        if child.tag != f"{W}p":
            continue
        if _paragraph_text(child).startswith(caption_prefix):
            if idx + 1 < len(children) and children[idx + 1].tag == f"{W}tbl":
                return children[idx + 1]
    return None


def fix_inserted_table_styles(report_path: Path = REPORT) -> None:
    with zipfile.ZipFile(report_path, "r") as zin:
        original_doc = zin.read("word/document.xml")
    root = ET.fromstring(original_doc)
    body = root.find(f"{W}body")
    if body is None:
        raise RuntimeError("document body not found")

    targets = [
        ("表 4 十帧转弯场景逐帧左右轮结果对比", 9),
        ("表 5 十帧 INT8+FP16 与 FP32 左右轮预测对比", 7),
    ]
    fixed = 0
    for prefix, ncols in targets:
        tbl = _find_table_after_caption(body, prefix)
        if tbl is None:
            continue
        _rebalance_table_grid(tbl, ncols)
        _apply_three_line_table(tbl)
        fixed += 1

    if fixed == 0:
        raise RuntimeError("inserted tables 4/5 not found")

    updated = _replace_document_body(original_doc, root)

    tmp = report_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(report_path, "r") as zin, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = updated
            zout.writestr(item, data)
    tmp.replace(report_path)
    print(f"Fixed table styles in {report_path} ({fixed} tables)")


def _replace_document_body(original: bytes, root: ET.Element) -> bytes:
    text = original.decode("utf-8")
    body = root.find(f"{W}body")
    if body is None:
        raise RuntimeError("document body not found")
    new_body = ET.tostring(body, encoding="unicode")
    updated, count = re.subn(r"<w:body>[\s\S]*?</w:body>", new_body, text, count=1)
    if count != 1:
        raise RuntimeError("failed to replace document body")
    return updated.encode("utf-8")


def _make_table(template_tbl: ET.Element, headers: list[str], rows: list[list[str]]) -> ET.Element:
    tbl = deepcopy(template_tbl)
    ncols = len(headers)

    grid = tbl.find(f"{W}tblGrid")
    if grid is None:
        grid = ET.Element(f"{W}tblGrid")
        tbl.insert(1, grid)
    for child in list(grid):
        grid.remove(child)
    col_w = str(max(800, 9000 // ncols))
    for _ in range(ncols):
        ET.SubElement(grid, f"{W}gridCol", {f"{W}w": col_w})

    proto_tr = template_tbl.findall(f"{W}tr")[0]
    proto_tc = deepcopy(proto_tr.findall(f"{W}tc")[0])
    _strip_w14_ids(proto_tc)
    proto_tc_pr = proto_tc.find(f"{W}tcPr")
    if proto_tc_pr is not None:
        for child in list(proto_tc_pr):
            if child.tag in (f"{W}tcBorders", f"{W}tcMar"):
                proto_tc_pr.remove(child)

    for tr in list(tbl.findall(f"{W}tr")):
        tbl.remove(tr)

    def make_row(values: list[str], *, header: bool = False) -> ET.Element:
        tr = ET.Element(f"{W}tr")
        tr_pr = proto_tr.find(f"{W}trPr")
        if tr_pr is not None:
            tr_pr_copy = deepcopy(tr_pr)
            _strip_w14_ids(tr_pr_copy)
            tr.insert(0, tr_pr_copy)
        for val in values:
            tc = deepcopy(proto_tc)
            _set_cell_text(tc, val)
            if header:
                for run in tc.iter(f"{W}r"):
                    rpr = run.find(f"{W}rPr")
                    if rpr is None:
                        rpr = ET.SubElement(run, f"{W}rPr")
                        run.insert(0, rpr)
                    if rpr.find(f"{W}b") is None:
                        ET.SubElement(rpr, f"{W}b")
            tr.append(tc)
        return tr

    tbl.append(make_row(headers, header=True))
    for row in rows:
        tbl.append(make_row(row))
    _rebalance_table_grid(tbl, len(headers))
    _apply_three_line_table(tbl)
    return tbl


def _build_rows_int8_vs_gt() -> tuple[list[str], list[list[str]]]:
    headers = [
        "帧",
        "真值左轮",
        "真值右轮",
        "预测左轮",
        "预测右轮",
        "左轮误差",
        "右轮误差",
        "轮速差",
        "转向",
    ]
    rows: list[list[str]] = []
    for frame in FRAMES:
        gt_l, gt_r, turn = GT[frame]
        pl, pr = INT8_FP16[frame]
        rows.append(
            [
                str(frame),
                _fmt(gt_l),
                _fmt(gt_r),
                _fmt(pl),
                _fmt(pr),
                _fmt(pl - gt_l),
                _fmt(pr - gt_r),
                _fmt(pl - pr),
                turn,
            ]
        )
    return headers, rows


def _build_rows_int8_vs_fp32() -> tuple[list[str], list[list[str]]]:
    headers = [
        "帧",
        "INT8+FP16 左",
        "INT8+FP16 右",
        "FP32 左",
        "FP32 右",
        "左轮差",
        "右轮差",
    ]
    rows: list[list[str]] = []
    for frame in FRAMES:
        i_l, i_r = INT8_FP16[frame]
        f_l, f_r = FP32[frame]
        rows.append(
            [
                str(frame),
                _fmt(i_l),
                _fmt(i_r),
                _fmt(f_l),
                _fmt(f_r),
                _fmt(i_l - f_l),
                _fmt(i_r - f_r),
            ]
        )
    return headers, rows


def _find_templates(root: ET.Element) -> tuple[ET.Element, ET.Element, ET.Element]:
    body = root.find(f"{W}body")
    body_paragraph = None
    caption_paragraph = None
    for child in body:
        if child.tag != f"{W}p":
            continue
        ppr = child.find(f"{W}pPr")
        if body_paragraph is None and ppr is not None and ppr.find(f"{W}ind") is not None:
            body_paragraph = child
        if caption_paragraph is None and _paragraph_text(child).startswith("表 3"):
            caption_paragraph = child
    tables = root.findall(f".//{W}tbl")
    template_tbl = tables[2] if len(tables) >= 3 else tables[0]
    if body_paragraph is None or caption_paragraph is None:
        raise RuntimeError("template paragraph not found")
    return body_paragraph, caption_paragraph, template_tbl


def _build_insert_block(root: ET.Element) -> str:
    body_p, caption_p, template_tbl = _find_templates(root)
    intro = _clone_paragraph(
        body_p,
        "为便于核对十帧转弯场景下的数值表现，下表给出 manifest 真值与选定部署模型（INT8+FP16）"
        "的逐帧左右轮速度对比。左轮误差、右轮误差分别为预测值减真值；十帧方向判断均与真值一致。",
    )
    caption1 = _clone_paragraph(caption_p, "表 4 十帧转弯场景逐帧左右轮结果对比（INT8+FP16）")
    h1, r1 = _build_rows_int8_vs_gt()
    table1 = _make_table(template_tbl, h1, r1)
    caption2 = _clone_paragraph(
        caption_p, "表 5 十帧 INT8+FP16 与 FP32 左右轮预测对比（QEMU）"
    )
    h2, r2 = _build_rows_int8_vs_fp32()
    table2 = _make_table(template_tbl, h2, r2)
    note = _clone_paragraph(
        body_p,
        "数据来源：QEMU 单帧推理日志（qemu_model_runs/QEMU_INFERENCE_REPORT.md），"
        "评测帧 37、59、228、229、313、331、392、463、542、586；每帧使用 manifest 真值 state。",
    )
    spacer = ET.Element(f"{W}p")
    return _serialize_fragment([intro, caption1, table1, caption2, table2, note, spacer])


def _insert_before_section4(doc: str, insert: str) -> str:
    pattern = (
        r"(<w:p\b[^>]*>(?:(?!</w:p>).)*?"
        r"<w:t[^>]*>四、模型方案对比</w:t>(?:(?!</w:p>).)*?</w:p>)"
    )
    match = re.search(pattern, doc, re.S)
    if not match:
        raise RuntimeError("anchor section 四 not found")
    pos = match.start()
    return doc[:pos] + insert + doc[pos:]


def _renumber_table_captions(doc: str, insert_end: int) -> str:
    tail = doc[insert_end:]
    for old, new in ((6, 8), (5, 7), (4, 6)):
        tail = re.sub(rf"(表\s+){old}(\s+)", rf"\g<1>{new}\2", tail)
    return doc[:insert_end] + tail


def update_report(report_path: Path = REPORT) -> None:
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    backup = report_path.with_suffix(".docx.bak")
    shutil.copy2(report_path, backup)

    with zipfile.ZipFile(report_path, "r") as zin:
        original_doc = zin.read("word/document.xml").decode("utf-8")
        root = ET.fromstring(original_doc.encode("utf-8"))
        insert = _build_insert_block(root)
        updated = _insert_before_section4(original_doc, insert)
        updated = _renumber_table_captions(updated, len(original_doc[: updated.find(insert)]) + len(insert))

        tmp = report_path.with_suffix(".tmp.docx")
        with zipfile.ZipFile(report_path, "r") as zin2, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED
        ) as zout:
            for item in zin2.infolist():
                data = zin2.read(item.filename)
                if item.filename == "word/document.xml":
                    data = updated.encode("utf-8")
                zout.writestr(item, data)
        tmp.replace(report_path)

    fix_inserted_table_styles(report_path)
    print(f"Updated {report_path}")
    print(f"Backup  {backup}")


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--fix-styles":
        fix_inserted_table_styles()
        return
    update_report()


if __name__ == "__main__":
    main()
