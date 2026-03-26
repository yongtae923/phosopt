"""Check NPZ file structure and integrity."""
import numpy as np
import zipfile
from pathlib import Path

print("=" * 70)
print("🔍 원본 파일 검증")
print("=" * 70)

orig_path = Path("data/letters/emnist_letters_phosphenes.npz")
filt_path = Path("data/letters/emnist_letters_phosphenes_filtered.npz")

# Check original
print("\n📌 원본 파일:")
try:
    with np.load(orig_path, allow_pickle=False) as data:
        for key in sorted(data.files):
            obj = data[key]
            if hasattr(obj, 'shape'):
                print(f"  ✅ {key:30} shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"  ⚠️  {key:30} type={type(obj).__name__}")
except Exception as e:
    print(f"  ❌ 오류: {e}")

# Check filtered with zipfile inspection
print("\n📌 필터링된 파일 (ZIP 구조):")
try:
    with zipfile.ZipFile(filt_path, 'r') as zf:
        print(f"  ZIP 내부 파일 목록:")
        for name in zf.namelist():
            info = zf.getinfo(name)
            print(f"    - {name:30} size={info.file_size:,} bytes")
except Exception as e:
    print(f"  ❌ 오류: {e}")

# Check filtered with np.load
print("\n📌 필터링된 파일 (NPZ 형식):")
try:
    with np.load(filt_path, allow_pickle=False) as data:
        print(f"  키 개수: {len(data.files)}")
        for key in sorted(data.files):
            try:
                obj = data[key]
                if hasattr(obj, 'shape'):
                    print(f"  ✅ {key:30} shape={obj.shape} dtype={obj.dtype}")
                else:
                    print(f"  ⚠️  {key:30} type={type(obj).__name__} (샘플 읽기 중...)")
                    # Try to actually load a sample
                    if key.endswith('phosphenes'):
                        _ = np.array(obj)
            except Exception as e:
                print(f"  ❌ {key:30} 읽기 실패: {str(e)[:50]}")
except Exception as e:
    print(f"  ❌ 오류: {e}")

print("\n" + "=" * 70)
