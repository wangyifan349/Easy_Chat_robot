#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from whoosh import index
from whoosh.fields import Schema, ID, TEXT
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.scoring import BM25F
# ---------- 配置 ----------
DOCS_DIR   = "docs"       # 文档目录，放 .txt 文件
INDEX_DIR  = "indexdir"   # whoosh 索引目录
PAGE_SIZE  = 10           # 每页显示条数
SCHEMA = Schema(
    path    = ID(stored=True, unique=True),
    title   = TEXT(stored=True, analyzer=StemmingAnalyzer(), field_boost=2.0),
    content = TEXT(stored=True, analyzer=StemmingAnalyzer(), field_boost=1.0),
)
# ---------- 工具函数 ----------
def clean_query(q: str) -> str:
    """去掉标点和特殊符号，只保留中英数字及空格"""
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9 ]+", " ", q).strip()
def ensure_docs_dir():
    if not os.path.isdir(DOCS_DIR):
        sys.stderr.write(f"Error: 找不到目录 '{DOCS_DIR}'，请先创建并放入 .txt 文件。\n")
        sys.exit(1)
def clear_index_dir():
    if os.path.exists(INDEX_DIR):
        for fn in os.listdir(INDEX_DIR):
            os.remove(os.path.join(INDEX_DIR, fn))
    else:
        os.makedirs(INDEX_DIR)
# ---------- 索引构建 ----------
def build_index():
    """
    构建或重建索引：读取 DOCS_DIR 下的所有 .txt 文件，建立 Whoosh 索引
    """
    ensure_docs_dir()
    clear_index_dir()
    ix = index.create_in(INDEX_DIR, SCHEMA)
    writer = ix.writer()
    files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".txt")]

    for fname in files:
        path = os.path.join(DOCS_DIR, fname)
        title = os.path.splitext(fname)[0]
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: 读取文件失败 {fname}: {e}")
            continue

        writer.add_document(path=path, title=title, content=content)
        print(f"Indexed: {fname}")

    writer.commit()
    print(f"索引完成，共 {len(files)} 个文档。")

# ---------- 搜索执行 ----------
def open_searcher():
    """打开已有索引并返回 searcher 对象和 parser 对象"""
    if not index.exists_in(INDEX_DIR):
        print("Error: 索引不存在，请先运行 `build` 命令。")
        return None, None

    ix = index.open_dir(INDEX_DIR)
    searcher = ix.searcher(weighting=BM25F(field_B={"title":2.0, "content":1.0}))
    parser = MultifieldParser(["title", "content"], schema=ix.schema, group=OrGroup)
    return searcher, parser

def do_search(raw_query: str, searcher, parser):
    """执行查询并返回所有结果列表"""
    q = clean_query(raw_query)
    if not q:
        return []

    query = parser.parse(q)
    results = searcher.search(query, limit=None)
    return list(results)
# ---------- 分页显示 ----------
def print_page(results, page):
    total = len(results)
    if total == 0:
        print("无结果。")
        return
    start = (page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    print(f"共 {total} 条结果，第 {page} 页（{start+1}–{end}）")
    for hit in results[start:end]:
        print(f"[{hit.score:.3f}] {hit['title']}  ({hit['path']})")
    print("-" * 40)
    print("输入 next/prev 翻页，build 重建索引，exit 退出。")
# ---------- 交互循环 ----------
def interactive_loop():
    searcher, parser = open_searcher()
    if not searcher:
        # 允许用户先 build
        pass
    results = []
    page = 1
    while True:
        try:
            cmd = input("查询> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not cmd:
            continue
        # 退出
        if cmd in ("exit", "quit"):
            print("Bye!")
            break
        # 重建索引
        if cmd == "build":
            build_index()
            searcher, parser = open_searcher()
            results = []
            page = 1
            continue
        # 翻页
        if cmd in ("next", "prev"):
            if not results:
                print("请先输入查询语句。")
                continue
            if cmd == "next":
                if page * PAGE_SIZE < len(results):
                    page += 1
                else:
                    print("已到最后一页。")
            else:  # prev
                if page > 1:
                    page -= 1
                else:
                    print("已到第一页。")

            print_page(results, page)
            continue
        # 普通查询
        if not searcher:
            # 如果还没索引，提示并继续
            print("请先运行 `build` 命令生成索引。")
            continue
        results = do_search(cmd, searcher, parser)
        page = 1
        print_page(results, page)
# ---------- 主入口 ----------
def main():
    # 支持命令行参数 build
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
        return

    print("命令：")
    print("  build       重建索引")
    print("  <关键字>    搜索（重建完索引后可用）")
    print("  next, prev  翻页")
    print("  exit, quit  退出")
    interactive_loop()

if __name__ == "__main__":
    main()
