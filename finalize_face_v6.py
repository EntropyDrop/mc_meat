"""Finish the authorized experiment after its existing training process ends."""
import json,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;run=ROOT/'runs/face_v6_latest_20261101'
launch=json.loads((ROOT/'diagnostics/face_v6_latest_20261101_launch.json').read_text())
while True:
 status=json.loads((run/'status.json').read_text())
 if status['stage']=='training_complete':break
 assert (Path('/proc')/str(launch['pid'])).exists(),'Training process exited before completion; inspect training log'
 time.sleep(5)
python=sys.executable
for script in ['report_face_v6.py','audit_face_v6.py']:
 subprocess.run([python,'-u',str(ROOT/script),'--run',str(run)],cwd=ROOT,check=True)
print('REPORT_AND_AUDIT_COMPLETE',flush=True)
