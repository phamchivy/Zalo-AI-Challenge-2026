"""Tiến trình con chạy code do LLM sinh. Gọi: python -I runner.py <pack.pkl> <code.py> <timeout_sec>

Code được cấp sẵn: pd, np, re, math, datetime/date/timedelta, json,
  tables : dict[tên bảng gốc -> DataFrame] (giá trị dạng chuỗi như trong file)
  canon  : dict['lines'|'orders'|'products'|'units'|'customers'|'payments'|'purchases' -> DataFrame] (đã chuẩn hoá)
  meta   : dict (as_of_date, data_min, data_max, business_id, ...)
  helpers: fold, to_num, parse_date, month_end, round_money
Code phải gán biến `result = {"type": ..., "answer": ..., "reason": ...}`.
"""
import json
import math
import os
import pickle
import re
import resource
import sys
import traceback
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def _jsonable(x):
    import numpy as np
    import pandas as pd
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (date, datetime, pd.Timestamp)):
        return x.isoformat()[:10]
    if isinstance(x, (set, tuple)):
        return list(x)
    if isinstance(x, pd.Series):
        return x.tolist()
    if isinstance(x, pd.DataFrame):
        return {'columns': list(x.columns), 'rows': x.values.tolist()}
    return str(x)


def main():
    pack_path, code_path, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 5, int(timeout) + 10))
        mem = int(os.environ.get('DATAMATE_SANDBOX_MEM_GB', '16')) * 1024 ** 3
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except Exception:  # noqa: BLE001
        pass
    import numpy as np
    import pandas as pd
    from datamate.core.textutil import fold, month_end, parse_date, to_num
    from datamate.reasoning.executor import round_money
    with open(pack_path, 'rb') as f:
        pack = pickle.load(f)
    code = open(code_path, encoding='utf-8').read()
    g = {'pd': pd, 'np': np, 're': re, 'math': math, 'json': json, 'date': date, 'datetime': datetime, 'timedelta': timedelta,
         'tables': pack['tables'], 'canon': pack['canon'], 'meta': pack['meta'],
         'fold': fold, 'to_num': to_num, 'parse_date': parse_date, 'month_end': month_end, 'round_money': round_money,
         '__name__': '__sandbox__'}
    out = {'status': 'ok'}
    try:
        exec(compile(code, '<llm_code>', 'exec'), g)
        res = g.get('result')
        if res is None and 'answer' in g:
            res = {'type': g.get('answer_type'), 'answer': g.get('answer')}
        if res is None:
            out = {'status': 'empty', 'error': 'code did not set `result`'}
        else:
            out['result'] = res
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        out = {'status': 'error', 'error': tb[-1500:]}
    sys.stdout.write('\n__RESULT__' + json.dumps(out, ensure_ascii=False, default=_jsonable) + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
