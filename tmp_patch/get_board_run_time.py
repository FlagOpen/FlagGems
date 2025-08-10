import os
import json
import csv

CASE_DIR = '/data/baai-benchmark-test-case'

KCORES = []
for root, dirs, files in os.walk(CASE_DIR):
    for filename in files:
        if filename == 'kcore_info.json':
            kcore_info_filepath = os.path.join(root, filename)
            KCORES.append({
                'kcore_info_filepath': kcore_info_filepath,
                'chip_out_path': os.path.join(os.path.dirname(kcore_info_filepath), 'chip_out', 'node_0_0/' )
                })


with open('result_new.csv', 'w', newline='') as result_file:
    fieldnames = ['op_name', 'dtype', 'shape_detail', 'latency_base', 'latency', 'latency_new', 'speedup_new', 'speedup', 'tflops']
    writer = csv.DictWriter(result_file, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()

    for kcore in KCORES:
        with open(kcore['kcore_info_filepath'], 'r') as f:
            case_info = json.load(f)
        f = os.popen(f'bash /data/baai-benchmark-test/run_engtest.sh {kcore["chip_out_path"]}')
        for line in f.readlines():
            if 'Run' in line:
                latency_new = float(line.split()[-1])/1000
                break
        case_info['latency_new'] = latency_new
        case_info['speedup_new'] = case_info['latency_base']/case_info['latency_new']
        writer.writerow(case_info)
