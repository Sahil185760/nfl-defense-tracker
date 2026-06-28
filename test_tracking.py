import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from defense_tracker import Tracker, detect, analyze_alignment, main

class TrackingTests(unittest.TestCase):
    def test_detect_and_region(self):
        image=np.full((120,200,3),(30,90,30),np.uint8)
        cv2.rectangle(image,(20,20),(35,45),(0,130,255),-1)
        cv2.rectangle(image,(140,20),(155,45),(0,130,255),-1)
        self.assertEqual(len(detect(image,'orange')),2)
        self.assertEqual(len(detect(image,'orange',roi=(0,0,100,120))),1)

    def test_continuity_gap_and_reset(self):
        tracker=Tracker()
        for i in range(5): tracks=tracker.update([(10+i*2,20,12,20)],i)
        self.assertEqual([t.id for t in tracks],[1])
        self.assertEqual(tracker.update([],5),[])
        tracks=tracker.update([(22,20,12,20)],6)
        self.assertEqual(tracks[0].id,1)
        tracker.reset()
        for i in range(7,10): tracks=tracker.update([(22,20,12,20)],i)
        self.assertEqual(tracks[0].id,2)

    def test_shell_is_not_coverage(self):
        rows=[{'frame':2,'id':i,'x':i*20,'y':y} for i,y in enumerate([10,10,80,95,98,99])]
        report=analyze_alignment(rows,30,100,5,'up',True)
        self.assertEqual(report['shell'],'Two-high shell candidate')
        self.assertEqual(report['classification'],'Alignment hypothesis only')
        self.assertEqual(analyze_alignment(rows,30)['classification'],'Undetermined')
        self.assertEqual(analyze_alignment([],30)['classification'],'Undetermined')

    def test_full_video_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'synthetic.avi';out=root/'result'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),15,(320,240))
            self.assertTrue(writer.isOpened())
            for i in range(30):
                image=np.full((240,320,3),(30,90,30),np.uint8)
                for x,y in [(35,30),(220,30),(70,150),(120,160),(180,150),(260,160)]:
                    cv2.rectangle(image,(x+i//3,y),(x+12+i//3,y+22),(0,130,255),-1)
                writer.write(image)
            writer.release()
            with contextlib.redirect_stdout(io.StringIO()):
                code=main([str(source),'--jersey','orange','--fixed-camera','--los','190','--pixels-per-yard','8','--output',str(out)])
            self.assertEqual(code,0)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['frames_processed'],30)
            self.assertEqual(report['track_count'],6)
            capture=cv2.VideoCapture(str(out/'tracked.mp4'))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),30)
            capture.release()
            self.assertIsNotNone(cv2.imread(str(out/'movement.png')))
            self.assertIn('Two-high',report['shell'])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(source),'--jersey','orange','--output',str(out)]),1)

if __name__=='__main__': unittest.main()
