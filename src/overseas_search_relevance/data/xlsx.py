"""只读 XLSX 解析器，用于在无 Office 依赖环境中审计内部工作簿。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_RE = re.compile(r"([A-Z]+)(\d+)")


def _column_index(reference: str) -> int:
    match = CELL_RE.fullmatch(reference)
    if match is None:
        raise ValueError(f"无效单元格引用：{reference}")
    result = 0
    for char in match.group(1):
        result = result * 26 + ord(char) - 64
    return result - 1


def iter_xlsx_sheets(path: Path) -> Iterator[tuple[str, list[list[str]]]]:
    """以字符串矩阵返回工作表；保留空列位置，不执行公式。"""

    ns = {"m": MAIN_NS, "r": REL_NS, "p": PKG_NS}
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{{{MAIN_NS}}}t")) for si in root]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
        sheets = workbook.find("m:sheets", ns)
        if sheets is None:
            return
        for sheet in sheets:
            name = sheet.attrib["name"]
            target = targets[sheet.attrib[f"{{{REL_NS}}}id"]]
            member = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            root = ET.fromstring(archive.read(member))
            output: list[list[str]] = []
            for row in root.findall(".//m:sheetData/m:row", ns):
                values: list[str] = []
                for sequential_index, cell in enumerate(row.findall("m:c", ns)):
                    # 少量 Office 导出文件会省略单元格引用，此时按当前行顺序恢复列位置。
                    reference = cell.attrib.get("r")
                    index = _column_index(reference) if reference else sequential_index
                    values.extend([""] * (index + 1 - len(values)))
                    value_node = cell.find("m:v", ns)
                    inline = cell.find("m:is", ns)
                    value = ""
                    if value_node is not None:
                        value = value_node.text or ""
                        if cell.attrib.get("t") == "s":
                            value = shared[int(value)]
                    elif inline is not None:
                        value = "".join(t.text or "" for t in inline.iter(f"{{{MAIN_NS}}}t"))
                    values[index] = value
                output.append(values)
            yield name, output
