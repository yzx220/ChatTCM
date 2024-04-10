import sys
sys.argv=['']
del sys
from api import cot
import os

question = "推荐一些缓解痰鸣的中药方剂"

A = cot(question=question)

print(A)