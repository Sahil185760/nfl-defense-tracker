"""Local color segmentation, conservative association, and alignment heuristics."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import cv2
import numpy as np

# OpenCV hue runs from 0 to 179. Thresholds are starting points, not learned weights.
PALETTES = {
    'white': [((0,0,165),(179,75,255))],
    'black': [((0,0,0),(179,255,65))],
    'blue': [((90,80,45),(130,255,255))],
    'red': [((0,90,60),(10,255,255)),((170,90,60),(179,255,255))],
    'orange': [((5,110,90),(25,255,255))],
    'yellow': [((20,80,100),(38,255,255))],
    'purple': [((125,65,35),(165,255,255))],
}


def detect(frame, jersey, min_area=30, max_area=4000, roi=None):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=np.zeros(frame.shape[:2],np.uint8)
    for lo,hi in PALETTES[jersey]:
        mask |= cv2.inRange(hsv,np.array(lo),np.array(hi))
    if roi:
        x,y,w,h=roi
        region=np.zeros_like(mask); region[y:y+h,x:x+w]=255
        mask &= region
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        area=cv2.contourArea(c)
        x,y,w,h=cv2.boundingRect(c)
        if min_area<=area<=max_area and .18<=w/max(1,h)<=1.8 and area/(w*h)>.22:
            boxes.append((x,y,w,h))
    return sorted(boxes,key=lambda b:(b[0],b[1]))


@dataclass
class Track:
    id: int
    box: tuple
    center: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda:np.zeros(2))
    missed: int = 0
    hits: int = 1
    history: list = field(default_factory=list)


class Tracker:
    """Motion-gated association. Ambiguous crossings produce gaps, not forced IDs."""
    def __init__(self,max_distance=60,max_missed=8):
        self.tracks=[]; self.next_id=1
        self.max_distance=max_distance; self.max_missed=max_missed

    def reset(self):
        self.tracks=[]  # Never reuse IDs across a cut.

    def update(self,boxes,frame_index):
        centers=[np.array([x+w/2,y+h/2]) for x,y,w,h in boxes]
        candidates=[]
        for ti,t in enumerate(self.tracks):
            predicted=t.center+t.velocity*(t.missed+1)
            ds=sorted((float(np.linalg.norm(c-predicted)),di) for di,c in enumerate(centers))
            if ds and ds[0][0]<=self.max_distance:
                # Near-equal candidates are not enough evidence for identity.
                if len(ds)>1 and ds[1][0]-ds[0][0]<5:
                    continue
                candidates.append((ds[0][0],ti,ds[0][1]))
        assigned_t=set();assigned_d=set()
        for _,ti,di in sorted(candidates):
            if ti in assigned_t or di in assigned_d: continue
            t=self.tracks[ti]
            t.velocity=(centers[di]-t.center)/(t.missed+1)
            t.center=centers[di];t.box=boxes[di];t.missed=0;t.hits+=1
            t.history.append((frame_index,*t.center.tolist()))
            assigned_t.add(ti);assigned_d.add(di)
        for ti,t in enumerate(self.tracks):
            if ti not in assigned_t: t.missed+=1
        self.tracks=[t for t in self.tracks if t.missed<=self.max_missed]
        for di,c in enumerate(centers):
            if di not in assigned_d:
                t=Track(self.next_id,boxes[di],c,history=[(frame_index,*c.tolist())])
                self.next_id+=1;self.tracks.append(t)
        return [t for t in self.tracks if t.missed==0 and t.hits>=3]


def analyze_alignment(rows,fps,los=None,pixels_per_yard=None,direction='up',fixed_camera=False):
    report={'classification':'Undetermined','front':'Unknown','shell':'Unknown','evidence':[],
        'limitations':['Color candidates can include officials, field paint, or sideline people.',
        'IDs may split or switch during occlusion; review the annotated video.',
        'Coverage calls, man/zone responsibilities, and blitz intent are not established by these rules.']}
    if not rows:
        report['limitations'].append('No confirmed tracks. Adjust jersey, region, or area thresholds.');return report
    if not fixed_camera or los is None or pixels_per_yard is None:
        report['limitations'].append('Alignment rules need --fixed-camera, --los, and --pixels-per-yard for a roughly overhead, field-aligned view.');return report
    report['limitations'].append('Depth is an approximate image-axis measurement, not perspective-correct field yardage.')
    first=min(r['frame'] for r in rows)
    # A stable early snapshot is needed; do not combine players from different times.
    snapshot=[r for r in rows if r['frame']==first]
    if len(snapshot)<5 or len(snapshot)>11:
        report['limitations'].append('Early snapshot has fewer than 5 or more than 11 candidates; shell inference withheld.');return report
    sign=-1 if direction=='up' else 1
    depth=lambda r:sign*(r['y']-los)/pixels_per_yard
    deep=[r for r in snapshot if depth(r)>=10]
    front=[r for r in snapshot if -1<=depth(r)<=2]
    report['front']=f'{len(front)} near-line candidates'
    report['shell']={0:'No deep candidate visible',1:'Single-high shell candidate',2:'Two-high shell candidate'}.get(len(deep),f'{len(deep)} deep candidates; shell uncertain')
    report['classification']='Alignment hypothesis only'
    report['evidence']=[f'{len(snapshot)} confirmed color tracks in first stable snapshot.',f'{len(deep)} candidates at least 10 approximate yards behind the specified line.',f'{len(front)} candidates within -1 to +2 approximate yards of the line.']
    by_id={}
    for r in rows: by_id.setdefault(r['id'],[]).append(r)
    advancing=[]
    for ident,path in by_id.items():
        if (path[-1]['frame']-path[0]['frame'])/fps>=.5 and depth(path[0])-depth(path[-1])>=2:
            advancing.append(ident)
    report['evidence'].append(f'{len(advancing)} tracks moved at least 2 approximate yards toward the offense over at least 0.5s; not a confirmed rush count.')
    return report

"""Terminal-only, offline football tracking pipeline."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np


def main(argv=None):
    p=argparse.ArgumentParser(description='Track defensive jersey candidates locally and export football movement diagrams. No cloud or API keys.')
    p.add_argument('video',type=Path)
    p.add_argument('--jersey',required=True,choices=PALETTES,help='Defensive jersey color')
    p.add_argument('--start',type=float,default=0)
    p.add_argument('--end',type=float,help='End time; default is 10 seconds after start')
    p.add_argument('--output',type=Path,help='New output folder; existing folders are not overwritten')
    p.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','W','H'),help='Restrict detection to field region in original pixels')
    p.add_argument('--min-area',type=int,default=30,help='Minimum jersey blob area in original pixels')
    p.add_argument('--max-area',type=int,default=4000)
    p.add_argument('--max-distance',type=float,default=60,help='Maximum association distance per frame in pixels')
    p.add_argument('--fixed-camera',action='store_true',help='Declare camera stationary; required for movement/shell rules')
    p.add_argument('--los',type=float,help='Horizontal line of scrimmage y-coordinate in original pixels')
    p.add_argument('--pixels-per-yard',type=float,help='Approximate vertical scale near the play')
    p.add_argument('--defense-direction',choices=['up','down'],default='up',help='Side of line occupied by defense in the image')
    args=p.parse_args(argv)
    cap=None;writer=None
    try:
        if not args.video.is_file(): raise ValueError(f'Video not found: {args.video}')
        if not math.isfinite(args.start) or args.start<0 or not 0<args.min_area<args.max_area or not math.isfinite(args.max_distance) or args.max_distance<=0: raise ValueError('Invalid start, area bounds, or tracking distance.')
        if args.pixels_per_yard is not None and (not math.isfinite(args.pixels_per_yard) or args.pixels_per_yard<=0): raise ValueError('Pixels per yard must be positive.')
        cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not math.isfinite(fps) or fps<=0 or count<1: raise ValueError('Cannot decode video. Try an H.264 MP4.')
        duration=count/fps;end=min(duration,args.start+10) if args.end is None else args.end
        if not math.isfinite(end) or not args.start<end<=duration+.001 or end-args.start>120: raise ValueError(f'Choose a range within {duration:.2f}s, up to 120 seconds long.')
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.los is not None and (not math.isfinite(args.los) or not 0<=args.los<h): raise ValueError('Line of scrimmage must be within image height.')
        if args.roi:
            x,y,rw,rh=args.roi
            if min(x,y)<0 or min(rw,rh)<=0 or x+rw>w or y+rh>h: raise ValueError('ROI must lie within the video.')
        out=args.output or Path('reports')/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        out.mkdir(parents=True,exist_ok=False)
        writer=cv2.VideoWriter(str(out/'tracked.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not writer.isOpened(): raise ValueError('Video encoder unavailable; install a supported OpenCV wheel.')
        start_frame=int(args.start*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        tracker=Tracker(args.max_distance);rows=[];first_image=None;previous=None;cuts=0;processed=0
        print('Tracking locally. Video stays on this computer.',flush=True)
        for index in range(start_frame,min(int(count),math.ceil(end*fps))):
            ok,image=cap.read()
            if not ok: raise ValueError(f'Decode failed at frame {index}; outputs may be incomplete.')
            if first_image is None: first_image=image.copy()
            gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(160,90))
            if previous is not None and np.mean(cv2.absdiff(previous,gray))>45:
                tracker.reset();cuts+=1
            previous=gray
            tracks=tracker.update(detect(image,args.jersey,args.min_area,args.max_area,args.roi),index)
            for t in tracks:
                x,y,bw,bh=t.box
                rows.append({'frame':index,'time':round(index/fps,4),'id':t.id,'x':round(float(t.center[0]),2),'y':round(float(t.center[1]),2)})
                cv2.rectangle(image,(x,y),(x+bw,y+bh),(80,240,180),2)
                cv2.putText(image,f'D{t.id}',(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(80,240,180),2)
                if args.fixed_camera and len(t.history)>1:
                    points=np.array([[int(a),int(b)] for _,a,b in t.history[-90:]],np.int32)
                    cv2.polylines(image,[points],False,(80,240,180),2)
            if args.los is not None: cv2.line(image,(0,round(args.los)),(w-1,round(args.los)),(0,230,255),2)
            cv2.putText(image,f'{index/fps:.2f}s | {len(tracks)} candidates',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            writer.write(image);processed+=1
            if processed%100==0: print(f'  {processed} frames processed',flush=True)
        writer.release();writer=None
        with (out/'tracks.csv').open('w',newline='',encoding='utf-8') as f:
            csvwriter=csv.DictWriter(f,fieldnames=['frame','time','id','x','y']);csvwriter.writeheader();csvwriter.writerows(rows)
        report=analyze_alignment(rows,fps,args.los,args.pixels_per_yard,args.defense_direction,args.fixed_camera and cuts==0)
        report.update({'clip':args.video.name,'jersey':args.jersey,'frames_processed':processed,'camera_cuts_detected':cuts,'track_count':len({r['id'] for r in rows}),'method':'Local HSV segmentation and motion-gated association'})
        if cuts: report['limitations'].append('Camera cut detected: identities reset and alignment inference withheld.')
        diagram=np.full((h,w,3),(35,55,35),np.uint8)
        if args.fixed_camera and cuts==0:
            diagram=cv2.addWeighted(first_image,.35,diagram,.65,0)
            for ident in sorted({r['id'] for r in rows}):
                path=[r for r in rows if r['id']==ident]
                points=np.array([[round(r['x']),round(r['y'])] for r in path],np.int32)
                color=(int(80+(ident*47)%175),int(100+(ident*71)%155),int(90+(ident*29)%165))
                if len(points)>1:
                    cv2.polylines(diagram,[points],False,color,2)
                    cv2.arrowedLine(diagram,tuple(points[-2]),tuple(points[-1]),color,2,tipLength=.5)
                cv2.putText(diagram,f'D{ident}',tuple(points[0]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
            if args.los is not None: cv2.line(diagram,(0,round(args.los)),(w-1,round(args.los)),(0,230,255),2)
            label=report['shell']+' | image-space paths'
        else:
            label='Movement diagram withheld: use a fixed camera clip without cuts.'
        cv2.putText(diagram,label[:100],(12,28),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        if not cv2.imwrite(str(out/'movement.png'),diagram): raise ValueError('Could not write diagram.')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        text='DEFENSE LAB\n'+report['classification']+'\nFront: '+report['front']+'\nShell: '+report['shell']+'\n\nEvidence\n'+'\n'.join('- '+x for x in report['evidence'])+'\n\nLimits\n'+'\n'.join('- '+x for x in report['limitations'])
        (out/'report.txt').write_text(text+'\n',encoding='utf-8')
        print(text);print(f'\nSaved annotated video, tracks, diagram and report to: {out.resolve()}')
        return 0
    except (ValueError,OSError) as e:
        print(f'Error: {e}',file=sys.stderr);return 1
    except KeyboardInterrupt:
        print('\nCanceled. Output may be incomplete.',file=sys.stderr);return 130
    finally:
        if cap is not None: cap.release()
        if writer is not None: writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
