import json
from schema_parser import load_schema
from schema_linker import SchemaLinker
from async_keyword_extractor import KeywordExtractor
from typing import Dict, Any
import os
import asyncio

TRAIN_JSON_PATH = r"/home/yangliu26/data/train/train.json"
SCHEMA_JSON_PATH = r"/home/yangliu26/data/train/train_tables.json"
MODEL_PATH = r"/home/yangliu26/qwen3-8b"

# 加载schema信息
def get_schema_map(schema_json_path: str) -> Dict[str, Any]:
    schema = load_schema(schema_json_path)
    db_schema_map = {}
    for db in schema:
        db_id = db["db_id"] if isinstance(db, dict) and "db_id" in db else db.get("db_id", "")
        db_schema_map[db_id] = db
    return db_schema_map

def link_keywords_to_schema(keywords, schema_info):
    # 简单schema linking逻辑：关键词与表名、字段名做模糊匹配
    linked = []
    tables = schema_info.get("table_names", [])
    columns = [col[1] for col in schema_info.get("column_names", [])]
    for kw in keywords:
        for t in tables:
            if kw.lower() in t.lower():
                linked.append((kw, "table", t))
        for c in columns:
            if kw.lower() in c.lower():
                linked.append((kw, "column", c))
    return linked

async def async_main():
    with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 取前10个数据作为测试
    data = data[:100]
    schema_map = get_schema_map(SCHEMA_JSON_PATH)
    print("schema_map提取完成")
    extractor = KeywordExtractor(MODEL_PATH)
    print("extractor建立完成")
    linker = SchemaLinker()
    print("linker建立完成")
    # 提取出所有问题
    questions = [sample["question"] for sample in data]
    print("问题提取完成")
    # 提取出每个问题的关键词
    print("开始提取关键词")
    all_keywords = await extractor.batch_extract(questions)
    print("提取关键词完成")
    results = []
    for idx, (sample, keywords) in enumerate(zip(data, all_keywords), 1):
        db_id = sample["db_id"]
        question = sample["question"]
        evidence = sample["evidence"]
        schema_info = schema_map.get(db_id, {})
        # schema-linking
        linker.build_index(schema_info)
        linking_results = linker.search(keywords)
        # 格式化
        formatted_linking = {}
        for matches in linking_results:
            for kw, schema_item, table_name, score in matches:
                if table_name not in formatted_linking:
                    formatted_linking[table_name] = []
                formatted_linking[table_name].append(schema_item)

        results.append({
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "keywords": keywords,
            "schema_linking": formatted_linking
        })

        # --- 每 10 条立即保存 ---
        if idx % 10 == 0:
            flush_path = os.path.join(os.path.dirname(__file__), "schema_linking_result.json")
            with open(flush_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已处理 {idx}/{len(data)}，中间结果写入 {flush_path}")
        
    # 输出结果到当前文件目录下的schema_linking_result.json
    out_path = os.path.join(os.path.dirname(__file__), "schema_linking_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Schema linking结果已保存到: {out_path}")

if __name__ == "__main__":
    asyncio.run(async_main())