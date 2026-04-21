import numpy as np
import zipfile

orig = 'data/letters/emnist_letters_phosphenes.npz'
filt = 'data/letters/emnist_letters_phosphenes_filtered.npz'

print('=== 원본 구조 ===')
with zipfile.ZipFile(orig, 'r') as zf:
    for name in zf.namelist():
        print(f'  {name}')

print('\n=== 필터링 구조 ===')
with zipfile.ZipFile(filt, 'r') as zf:
    for name in zf.namelist():
        print(f'  {name}')

print('\n=== numpy 로드 테스트 ===')
try:
    with np.load(orig, allow_pickle=False) as data:
        print('원본:')
        for k in data.keys():
            obj = data[k]
            print(f'  {k}: type={type(obj)}, has_shape={hasattr(obj, "shape")}')
except Exception as e:
    print(f'원본 오류: {e}')

try:
    with np.load(filt, allow_pickle=False) as data:
        print('필터링:')
        for k in data.keys():
            obj = data[k]
            print(f'  {k}: type={type(obj)}, has_shape={hasattr(obj, "shape")}')
except Exception as e:
    print(f'필터링 오류: {e}')