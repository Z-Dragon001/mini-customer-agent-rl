"""Finish one owned evaluation before replacing its paused five-arm scheduler."""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import run_retail_grpo_repair as experiment


def process(pid,start_ticks):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    if fields[19]!=start_ticks:
        raise RuntimeError('Process identity changed; refusing to signal a reused PID')
    return fields


def main():
    root,out=experiment.ROOT,experiment.OUT
    path=out/'attention_controller_pause.json'
    state=experiment.read(path)
    if state.get('closed'):
        raise RuntimeError('This scheduler handoff has already completed')
    parent=state['parent'];child=state['child']
    if process(parent,state['parent_start_ticks'])[0]!='T':
        raise RuntimeError('Expected the owned scheduler to be paused')
    while True:
        status=process(child['pid'],child['start_ticks'])
        if status[0]=='Z':
            if int(status[49])!=0:
                os.kill(parent,signal.SIGCONT)
                raise RuntimeError('Evaluation child failed; resumed original scheduler for recovery')
            break
        gpu=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True)
        if gpu.returncode==0:
            with (out/'gpu.csv').open('a') as handle:
                handle.write(f'{datetime.now(timezone.utc).isoformat()},progress-dev43,{gpu.stdout.strip()}\n')
        time.sleep(5)
    folder=root/experiment.read(out/'runs.json')['progress-dev43']
    experiment.valid_rows(folder)
    current=experiment.read(out/'status.json')
    experiment.write(folder/'phase-complete.json',{'phase':'progress-dev43','log':current['log'],
                      'note':'Verified child exit status 0 while its scheduler was paused'})
    experiment.event('progress-dev43','finished',returncode=0,folder=str(folder.relative_to(root)),
                     note='No evaluation interruption; completed before scheduler handoff')
    process(parent,state['parent_start_ticks'])
    os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
    state['closed']=True
    experiment.write(path,state)
    subprocess.run([sys.executable,'scripts/run_retail_grpo_repair.py','--dev-only'],cwd=root,check=True)
    subprocess.run([sys.executable,'scripts/run_retail_attention_repair.py'],cwd=root,check=True)


if __name__=='__main__':main()
