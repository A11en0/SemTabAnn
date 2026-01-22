import os
import re

def extract_f1_from_log(file_path, tail_lines=50):
    """Extract Micro/Macro F1 from the end of a single log file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[-tail_lines:]
    text = ''.join(lines)
    micro = re.search(r'Micro\s*F1:\s*([0-9.]+)', text)
    macro = re.search(r'Macro\s*F1:\s*([0-9.]+)', text)
    return float(micro.group(1)) if micro else None, float(macro.group(1)) if macro else None

def scan_directory(root_dir, suffix=".log"):
    """Recursively scan directory and extract results from each log file"""
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(suffix):
                fpath = os.path.join(dirpath, fname)
                micro, macro = extract_f1_from_log(fpath)
                if micro is not None or macro is not None:
                    results.append({
                        "file": fpath,
                        "micro_f1": micro,
                        "macro_f1": macro
                    })
    return results

if __name__ == "__main__":
    root = "outputs/igd_search_EM_1027"
    results = scan_directory(root)
    print(f"{'File':70s} | {'Micro-F1':>8s} | {'Macro-F1':>8s}")
    print("-"*95)
    for r in sorted(results, key=lambda x: x['micro_f1'] or 0, reverse=True):
        print(f"{r['file'][:70]:70s} | {r['micro_f1'] or '-':>8} | {r['macro_f1'] or '-':>8}")
        