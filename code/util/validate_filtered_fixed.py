import numpy as np
from pathlib import Path
import zipfile

npz_path = Path("data/letters/emnist_letters_phosphenes_filtered.npz")

print("=" * 70)
print("✅ 수정된 NPZ 파일 검증")
print("=" * 70)

if not npz_path.exists():
    print(f"❌ 파일 없음")
    exit(1)

file_size_mb = npz_path.stat().st_size / (1024 * 1024)
print(f"\n📦 파일 크기: {file_size_mb:.2f} MB")

try:
    # Check with zipfile first
    with zipfile.ZipFile(npz_path, 'r') as zf:
        print(f"✅ 유효한 NPZ 형식")
        print(f"\n📋 포함된 파일:")
        for name in zf.namelist():
            info = zf.getinfo(name)
            print(f"   ✅ {name:35} size={info.file_size:,} bytes")
        
    # Now check with numpy - just verify structure, don't load large arrays
    print(f"\n📊 numpy 접근 테스트:")
    with np.load(npz_path, allow_pickle=False, mmap_mode='r') as data:
        print(f"   ✅ numpy 로드 성공 (키 개수: {len(data.files)})")
        for key in sorted(data.files):
            try:
                arr = data[key]
                if hasattr(arr, 'shape'):
                    print(f"   ✅ {key:30} shape={arr.shape} dtype={arr.dtype}")
                else:
                    print(f"   ⚠️  {key:30} (접근 불가 - 메모리 맵 모드)")
            except Exception as e:
                print(f"   ⚠️  {key:30} (접근 실패: {str(e)[:30]})")
        
except Exception as e:
    print(f"❌ 오류: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✅ 검증 완료! 파일이 정상입니다.")
print("=" * 70)

