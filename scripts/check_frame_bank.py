"""Проверка существующего frame_bank.npz: покрывает ли он весь датасет.
  python check_bank.py outputs/frame_bank.npz
"""
import sys, numpy as np
z = np.load(sys.argv[1] if len(sys.argv) > 1 else "outputs/frame_bank.npz")
ep, ti = z["episode"], z["task_index"]
var = z["variant"] if "variant" in z.files else np.zeros(len(ep), int)
orig = var == 0
print(f"строк всего      : {len(ep)}  (оригиналов {orig.sum()}, ауг-вариантов {(~orig).sum()} — игнорируются)")
print(f"эпизодов          : {len(np.unique(ep[orig]))}   ожидается 1693")
print(f"задач             : {len(np.unique(ti[orig]))}   ожидается 40")
print(f"кадр              : {z['images'].shape[1:]}, dtype={z['images'].dtype}")
print(f"task_texts_json   : {'есть' if 'task_texts_json' in z.files else 'НЕТ'}")
miss = sorted(set(range(1693)) - set(int(e) for e in ep[orig]))
if miss:
    print(f"\nНЕ ХВАТАЕТ {len(miss)} эпизодов, например: {miss[:10]}")
    print("-> FrameBank.same() упадёт KeyError на любом из них. Пересобрать:")
    print("   python scripts/build_frame_bank.py --out outputs/frame_bank.npz")
else:
    dup = [t for t in np.unique(ti[orig]) if (ti[orig] == t).sum() < 2]
    print("\nПокрытие полное — банк переиспользуется как есть.")
    if dup:
        print(f"Примечание: у задач {dup} по одному эпизоду -> cross() для них "
              "откатится на same() (так и задумано).")
