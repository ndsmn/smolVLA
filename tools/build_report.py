"""Build the two-page training and evaluation report from saved metrics."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

P=Path(__file__).resolve().parents[1]/"docs";OUT=P/"output/pdf";A=P/"assets"
NAVY=HexColor("#0c1422");INK=HexColor("#172b42");MUTED=HexColor("#52677d");GREEN=HexColor("#167c60");LIGHT=HexColor("#eef5f8")
pdfmetrics.registerFont(TTFont("D","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DB","/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
W,H=A4


def line(c,x,y,text,size=10,color=INK,bold=False):
    c.setFillColor(color);c.setFont("DB" if bold else "D",size);c.drawString(x,y,text)
def para(c,text,x,top,width,size=10.1,leading=14,color=INK):
    style=ParagraphStyle("p",fontName="D",fontSize=size,leading=leading,textColor=color)
    p=Paragraph(text,style);_,h=p.wrap(width,900);p.drawOn(c,x,top-h);return h
def footer(c,page):
    c.setStrokeColor(HexColor("#d5e0e8"));c.line(40,48,W-40,48)
    line(c,40,31,"SO-101 / SmolVLA  /  MuJoCo evaluation",8,MUTED)
    line(c,W-73,31,f"{page} / 2",8,MUTED)


def main():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/"so101_smolvla_report.pdf"
    c=canvas.Canvas(str(path),pagesize=A4);c.setTitle("SmolVLA on the SO-101 - Training and Evaluation");c.setAuthor("ndsmn")
    c.setFillColor(NAVY);c.rect(0,650,W,H-650,fill=1,stroke=0)
    line(c,40,805,"SO-101 MANIPULATION  /  2026",10,HexColor("#71e4ad"),True)
    line(c,40,759,"SmolVLA",34,white,True);line(c,40,715,"on the SO-101",34,white,True)
    line(c,40,680,"Evaluating SmolVLA manipulation on a simulated SO-101 arm",10.5,HexColor("#c4d2df"))
    c.drawImage(str(A/"camera_alignment.png"),40,535,width=W-80,height=94,preserveAspectRatio=True,anchor="c")
    line(c,40,523,"Source / replica pairs: front, wrist and overhead views",8.4,MUTED)
    cards=[("40k","archived fine-tuning updates"),("78.6%","lower held-out action MAE"),("200","evaluated policy rollouts")]
    cw=(W-104)/3
    for i,(value,label) in enumerate(cards):
        x=40+i*(cw+12);c.setFillColor(LIGHT);c.roundRect(x,448,cw,61,8,fill=1,stroke=0)
        line(c,x+13,479,value,24,GREEN,True);line(c,x+13,460,label,7.7,MUTED)
    line(c,40,427,"TASK",9,GREEN,True)
    para(c,"Evaluate how a fine-tuned vision-language-action policy behaves in a reconstructed robot environment, and identify the gaps between demonstration prediction and completed pick-and-place tasks.",40,414,W-80,10.2,14)
    line(c,40,357,"IMPLEMENTATION",9,GREEN,True)
    bullets=["Adapted a published MuJoCo replica with three camera views, six named joint channels, seeded resets and contact-based grasping.",
        "Integrated the archived 40k-step SmolVLA checkpoint and its saved processors; ran frozen inference on a 6 GB RTX 3060.",
        "Built stage-level evaluation, paired experiments, confidence intervals, per-seed physical traces and success/failure video evidence."]
    top=342
    for b in bullets:
        line(c,41,top-10,"-",11,GREEN,True);h=para(c,b,53,top,W-93,9.7,13);top-=h+9
    line(c,40,224,"CLOSED-LOOP CONTROL",9,GREEN,True)
    boxes=[("OBSERVE","3 RGB views\n6 joint angles"),("PREDICT","SmolVLA\n50 joint targets"),("EXECUTE","25 targets\nMuJoCo physics"),("EVALUATE","Task stages\nStable placement")]
    bw=(W-119)/4
    for i,(title,body) in enumerate(boxes):
        x=40+i*(bw+13);c.setFillColor(NAVY);c.roundRect(x,142,bw,66,6,fill=1,stroke=0)
        line(c,x+9,190,title,8,HexColor("#71e4ad"),True)
        for j,s in enumerate(body.split("\n")):line(c,x+9,171-j*13,s,8.3,white)
        if i<3:line(c,x+bw+3,169,">",10,MUTED,True)
    para(c,"New observations after each executed chunk. Control period: 34 ms. Physics stepping is serialized; the simulator waits while the policy predicts.",40,128,W-80,8.5,11.5,MUTED)
    line(c,40,86,"STACK  Python / PyTorch / LeRobot / MuJoCo / Gymnasium",8.5,INK,True)
    para(c,"Training: Threadripper PRO / RTX 6000 Ada 48 GB. Inference: RTX 3060 6 GB. Offline action MAE uses 128 held-out trajectory anchors; it measures prediction error rather than task completion.",40,74,W-80,7.3,9,MUTED)
    footer(c,1);c.showPage()
    c.setFillColor(NAVY);c.rect(0,746,W,H-746,fill=1,stroke=0)
    line(c,40,803,"EVALUATION RESULTS",23,white,True)
    line(c,40,771,"Same 40k weights / 50 paired starts / original anti-aliased rendering",8.5,HexColor("#c4d2df"))
    c.drawImage(str(A/"success_comparison.png"),40,494,width=W-80,height=236)
    c.drawImage(str(A/"task_stages.png"),40,278,width=W-80,height=212)
    line(c,40,259,"FINDINGS",9,GREEN,True)
    notes=["The source-fit candidate reached the cube in 50/50 trials, lifted it in 41/50 (82%) and completed strict placement in 21/50 (42%). Placement remains the main bottleneck.",
        "Baseline placement was 17/50 (34%). The observed +8 percentage points are inconclusive: paired exact p = 0.4545. Candidate 95% Wilson interval: 29.4-55.8%.",
        "A zero-tolerance containment defect undercounted settled cubes. Exact replays of 100 old traces verified a 0.1 mm contact-tolerance correction; it changes measurement, not actions."]
    top=246
    for n in notes:
        h=para(c,n,40,top,W-80,8.9,11.8);top-=h+7
    para(c,"<b>Scope:</b> Simulation-only evaluation in a published replica; original contact physics remain uncertain. Material selection used one source episode. No real-robot validation or statistically established material improvement is claimed.",40,119,W-80,8.2,10.5,MUTED)
    line(c,40,75,"FILES  demo.mp4 / per-trial clips / JSON and CSV outcomes / source hashes",7.7,INK,True)
    line(c,40,60,"CREDITS  Hugging Face SmolVLA/LeRobot; gpudad dataset; Rovo Labs scene (Apache-2.0).",7.2,MUTED)
    footer(c,2);c.save();print(path,flush=True)


if __name__=="__main__":main()
