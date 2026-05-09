import asyncio
import pandas as pd
import io
import json


# Mimic the tool implementation from retrieve.py
async def parse_markdown_table(md_table: str, value: str):
    """
    解析 Markdown 格式的表格，并获取包含指定value的数据行。
    参数:
    - md_table: 原始 Markdown 格式的表格文本。以 '|' 分隔。
    - value: 需要查找的值
    """
    try:
        # 1. 预处理：去掉开头和结尾的 '|'，避免 Pandas 生成多余的空列
        cleaned_lines = []
        for line in md_table.strip().split("\n"):
            line = line.strip()
            if line.startswith("|"):
                line = line[1:]
            if line.endswith("|"):
                line = line[:-1]
            if line:
                cleaned_lines.append(line)

        # 2. 重构为 CSV 格式，Pandas 解析更准确
        # 跳过第二行（--- 分隔符）
        csv_data = "\n".join([line for line in cleaned_lines if "---" not in line])

        df = pd.read_csv(io.StringIO(csv_data), sep="|", skipinitialspace=True)

        # 3. 清理列名（去除可能的空格）
        df.columns = df.columns.str.strip()
        # 清理所有单元格数据（去除两侧空格）
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        data = df.to_dict(orient="records")

        # 4. 筛选逻辑
        results = [
            row
            for row in data
            if any(str(value) in str(cell_value) for cell_value in row.values())
        ]

        return results
    except Exception as e:
        return f"解析表格出错: {str(e)}"


async def main():
    # Sample markdown table
    md_table = """
||**设备**<br>**月份**|**6633#**|||||||**6634#**||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||设备<br>月份|100m|90m|80m|70m|50m|30m|20m|90m|80m|60m|30m|
||2018<br>年7<br>月|6.69|7.05|6.88|6.67|6.74|6.47|6.34|7.01|6.88|6.82|6.63|
||2018<br>年8<br>月|5.49|5.50|5.42|5.39|5.17|5.11|5.00|5.44|5.47|5.26|5.05|
||2018<br>年9<br>月|6.38|6.21|6.27|6.29|6.16|5.89|5.62|6.47|6.21|6.31|6.16|
||2018<br>年10<br>月|8.87|8.42|8.48|8.91|8.93|8.46|8.18|9.68|9.22|9.58|9.55|
||2018<br>年11<br>月|10.31|9.80|9.81|9.99|10.38|9.75|9.53|10.35|9.68|10.33|10.34|
||2018<br>年12<br>月|10.61|9.02|9.87|10.12|10.52|9.78|9.35|10.53|9.81|10.18|10.48|
||2019<br>年1<br>月|10.12|9.09|9.03|9.45|9.90|9.16|8.83|10.44|9.16|7.97|9.90|
||2019<br>年2<br>月|8.76|8.92|7.05|7.80|8.64|8.14|7.88|8.91|7.07|8.72|8.45|
||2019<br>年3<br>月|8.75|8.90|7.83|7.67|7.39|7.89|7.80|8.89|7.85|8.44|8.13|
||2019<br>年4<br>月|7.34|6.68|6.78|6.04|0.17|6.26|6.08|6.68|6.80|6.45|5.35|
||2019<br>年5<br>月|8.57|8.32|7.90|7.53|2.22|7.27|6.90|8.33|8.01|7.38|7.75|
||2019<br>年6<br>月|6.25|5.37|5.85|5.22|3.39|4.92|4.09|6.08|5.96|3.58|5.23|
||2019<br>年7<br>月|7.25|7.18|6.96|6.93|6.26|5.70|5.45|7.27|7.10|6.78|5.87|
||2019<br>年8<br>月|7.79|7.72|7.57|7.60|7.04|6.72|6.52|7.77|7.65|7.40|6.74|
||2019<br>年9<br>月|7.31|7.24|7.33|7.43|7.18|7.23|7.05|7.28|6.18|7.28|7.06|
||2019<br>年10<br>月|9.31|9.21|9.45|9.50|9.26|9.29|8.94|9.25|9.30|9.20|9.10|
||||||||||||||
"""

    print("--- Test 1: Finding a specific month ---")
    res1 = await parse_markdown_table(md_table, "2019<br>年5<br>月")
    print(
        f"Query: '2019年5月'\nResult: {json.dumps(res1, indent=4, ensure_ascii=False)}"
    )

    print("\n--- Test 2: Finding a value in the middle ---")
    res2 = await parse_markdown_table(md_table, "7.45")
    print(f"Query: '7.45'\nResult: {json.dumps(res2, indent=4, ensure_ascii=False)}")

    print("\n--- Test 3: Value not found ---")
    res3 = await parse_markdown_table(md_table, "2020年")
    print(f"Query: '2020年'\nResult: {json.dumps(res3, indent=4, ensure_ascii=False)}")

    print("\n--- Test 4: Partial match ---")
    res4 = await parse_markdown_table(md_table, "8.12")
    print(f"Query: '8.12'\nResult: {json.dumps(res4, indent=4, ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(main())
