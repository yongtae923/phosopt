import numpy as np
import zipfile
from pathlib import Path

npz_path = Path("D:\\yongtae\\phosopt\\data\\letters\\emnist_letters_phosphenes_filtered.npz")

print("=" * 70)
print("📦 NPZ 파일 검증")
print("=" * 70)

# 1. 파일 존재 및 기본 정보
if not npz_path.exists():
    print(f"❌ 파일 없음: {npz_path}")
    exit(1)

file_size_mb = npz_path.stat().st_size / (1024 * 1024)
print(f"✅ 파일 크기: {file_size_mb:.2f} MB")

# 2. NPZ 유효성 검사
try:
    with np.load(npz_path, allow_pickle=False) as data:
        keys = sorted(data.files)
        print(f"✅ 유효한 NPZ 형식 (키 개수: {len(keys)})")
        print(f"\n📋 포함된 배열:")
        
        for key in keys:
            arr = data[key]
            if isinstance(arr, np.ndarray):
                print(f"   - {key:30} shape={str(arr.shape):20} dtype={arr.dtype}")
            else:
                print(f"   - {key:30} type={type(arr).__name__:20} (not an array)")
            
        print(f"\n📊 데이터 통계:")
        
        # Train/test 포스펜 데이터 검증
        for split in ["train", "test"]:
            phos_key = f"{split}_phosphenes"
            if phos_key in data.files:
                phos = data[phos_key]
                print(f"\n   {split.upper()} Phosphenes:")
                print(f"      - 샘플 수: {phos.shape[0]}")
                print(f"      - 이미지 크기: {phos.shape[-2]}×{phos.shape[-1]}")
                print(f"      - 최소값: {phos.min():.6f}")
                print(f"      - 최대값: {phos.max():.6f}")
                print(f"      - 평균값: {phos.mean():.6f}")
                print(f"      - 0이 아닌 픽셀: {(phos > 0).mean():.2%}")
                
            label_key = f"{split}_labels"
            if label_key in data.files:
                labels = data[label_key]
                print(f"      - Labels 크기: {labels.shape}")
                
except zipfile.BadZipFile:
    print(f"❌ 손상된 파일 (유효하지 않은 ZIP)")
    exit(1)
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✅ 파일 검증 완료!")
print("=" * 70)
