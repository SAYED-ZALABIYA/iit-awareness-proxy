# iit_binary_causal_demo.py
# Minimal binary-gates causal system with a true TPM + PyPhi Φ.
# Nodes are binary, synchronous update, deterministic.
# PyPhi is used on the full system (no MLP, no proxies).

import itertools
import argparse
import numpy as np

try:
    import pyphi
except ImportError as e:
    raise SystemExit("PyPhi غير مثبت. ثبّت أولاً: pip install pyphi") from e


# ---------- تعريف بوابات منطقية ثنائية (قابلة للتبديل) ----------
def ID(x):        return x
def NOT(x):       return 1 - x
def AND(a, b):    return a & b
def OR(a, b):     return a | b
def XOR(a, b):    return a ^ b
def NAND(a, b):   return 1 - (a & b)
def NOR(a, b):    return 1 - (a | b)

# ---------- مولد شبكة صغيرة 3–5 عقد ----------
def make_network_spec(preset: str):
    """
    تعيد:
      funcs: قائمة دوال تحديث لكل عقدة على شكل lambda(prev_state)->0/1
      cm:    مصفوفة اتصال سببية (n×n) حيث cm[i,j]=1 يعني j يؤثر على i
      labels: أسماء العقد
    """
    preset = preset.lower()
    if preset == "and_fan_in_3":  # 3 عقد: A,B مدخلان، C = AND(A,B)، و A'=A, B'=B
        labels = ["A", "B", "C"]
        def fA(s): return ID(s[0])
        def fB(s): return ID(s[1])
        def fC(s): return AND(s[0], s[1])
        funcs  = [fA, fB, fC]
        cm = np.array([
            [1,0,0],  # A depends on A
            [0,1,0],  # B depends on B
            [1,1,0],  # C depends on A,B
        ], dtype=int)

    elif preset == "xor_with_memory_3":  # C = XOR(A,B), و C له وصلة ذاتية ضعيفة منطقياً: C'=XOR(A,B)
        labels = ["A", "B", "C"]
        def fA(s): return ID(s[0])
        def fB(s): return ID(s[1])
        def fC(s): return XOR(s[0], s[1])
        funcs  = [fA, fB, fC]
        cm = np.array([
            [1,0,0],
            [0,1,0],
            [1,1,0],  # C يتأثر بـ A,B
        ], dtype=int)

    elif preset == "and_or_4":  # 4 عقد: C=AND(A,B), D=OR(B,C). A,B ذاكرة هوية.
        labels = ["A", "B", "C", "D"]
        def fA(s): return ID(s[0])
        def fB(s): return ID(s[1])
        def fC(s): return AND(s[0], s[1])     # AND(A,B)
        def fD(s): return OR(s[1], s[2])      # OR(B,C)
        funcs  = [fA, fB, fC, fD]
        cm = np.array([
            [1,0,0,0],  # A<-A
            [0,1,0,0],  # B<-B
            [1,1,0,0],  # C<-A,B
            [0,1,1,0],  # D<-B,C
        ], dtype=int)

    elif preset == "xor_nand_5":  # 5 عقد: C=XOR(A,B), D=NAND(B,C), E=AND(C,D). A,B ذاكرة.
        labels = ["A", "B", "C", "D", "E"]
        def fA(s): return ID(s[0])
        def fB(s): return ID(s[1])
        def fC(s): return XOR(s[0], s[1])        # XOR(A,B)
        def fD(s): return NAND(s[1], s[2])       # NAND(B,C)
        def fE(s): return AND(s[2], s[3])        # AND(C,D)
        funcs  = [fA, fB, fC, fD, fE]
        cm = np.array([
            [1,0,0,0,0],  # A<-A
            [0,1,0,0,0],  # B<-B
            [1,1,0,0,0],  # C<-A,B
            [0,1,1,0,0],  # D<-B,C
            [0,0,1,1,0],  # E<-C,D
        ], dtype=int)
    else:
        raise ValueError("Preset غير معروف")

    return funcs, cm, labels


# ---------- بناء TPM على شكل "state-by-node" (احتمال node=1 | الحالة السابقة) ----------
def enumerate_states(n):
    return list(itertools.product([0,1], repeat=n))

def build_tpm_state_by_node(funcs, n):
    """
    تعطي مصفوفة (2**n × n) حيث كل قيمة ∈{0,1} تمثل P(node_i=1 | past_state).
    هذا شكل مقبول لـ pyphi.Network.
    """
    states = enumerate_states(n)
    tpm = np.zeros((2**n, n), dtype=float)
    for idx, s in enumerate(states):
        s = tuple(int(x) for x in s)
        nxt = [f(s) for f in funcs]         # تحديث متزامن
        tpm[idx, :] = np.array(nxt, dtype=float)  # احتمالات 1 (حتمي ⇒ 0/1)
    return tpm

# ---------- حساب Φ باستخدام PyPhi ----------
def compute_phi_for_state(tpm, cm, labels, state_tuple):
    net = pyphi.Network(tpm, connectivity_matrix=cm, node_labels=labels)
    sub = pyphi.Subsystem(net, state_tuple, tuple(range(len(labels))))
    phi_value = pyphi.compute.phi(sub)
    mip = pyphi.compute.sia(sub)  # يشتمل على MIP ومقاييس سببية
    return float(phi_value), mip


def main():
    p = argparse.ArgumentParser(description="Causal IIT demo with true binary TPM + PyPhi Φ")
    p.add_argument("--preset", choices=["and_fan_in_3","xor_with_memory_3","and_or_4","xor_nand_5"],
                   default="and_fan_in_3")
    p.add_argument("--state", type=str, default=None,
                   help="حالة ثنائية مثل 101 لأجل n عقد. إن لم تُحدد سيتم حساب Φ لجميع الحالات.")
    p.add_argument("--show_mip", action="store_true", help="أعرض تفاصيل الـMIP الأساسية.")
    args = p.parse_args()

    funcs, cm, labels = make_network_spec(args.preset)
    n = len(labels)
    tpm = build_tpm_state_by_node(funcs, n)

    if args.state is not None:
        if len(args.state) != n or any(c not in "01" for c in args.state):
            raise SystemExit(f"state يجب أن يكون بطول {n} ومكوّنًا من 0/1 فقط.")
        s = tuple(int(c) for c in args.state)
        phi, mip = compute_phi_for_state(tpm, cm, labels, s)
        print(f"Preset={args.preset} | labels={labels} | state={s} | Φ={phi:.6f}")
        if args.show_mip:
            print("MIP partition:", mip.cut)  # قد تغيّر صيغة العرض حسب نسخة PyPhi
    else:
        # مسح كل الحالات
        results = []
        for s in enumerate_states(n):
            phi, _ = compute_phi_for_state(tpm, cm, labels, s)
            results.append((s, phi))
        # عرض مرتبًا تنازليًا
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Preset={args.preset} | labels={labels}")
        for s, phi in results:
            print(f"state={s} -> Φ={phi:.6f}")


if __name__ == "__main__":
    main()
