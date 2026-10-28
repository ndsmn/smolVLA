"""Assemble the captioned simulation demo from recorded rollout clips."""
from pathlib import Path
import json
import math
import numpy as np
import av
from PIL import Image,ImageDraw,ImageFont,ImageEnhance

P=Path(__file__).resolve().parents[1]/"docs";A=P/"assets";W,H=1280,720;FPS=30
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";BOLD="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG="#0c1422";WHITE="#edf3fa";MUTED="#a8bacb";GREEN="#71e4ad";BLUE="#78aee8"


def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)
def text(im,xy,s,size=24,color=WHITE,bold=False):ImageDraw.Draw(im).text(xy,s,font=font(size,bold),fill=color)
def base(kicker,title):
    im=Image.new("RGB",(W,H),BG);d=ImageDraw.Draw(im)
    d.rectangle((44,38,86,43),fill=GREEN);text(im,(101,24),kicker,16,MUTED)
    text(im,(44,65),title,36,WHITE,True);return im
def load_clip(path):
    with av.open(str(path)) as c:return [Image.fromarray(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]
def fit(im,size):
    copy=im.copy();copy.thumbnail(size,Image.Resampling.LANCZOS);return copy


def main():
    manifest=json.loads((A/"proof/manifest.json").read_text());items={r["role"]:r for r in manifest["clips"]}
    clips={k:load_clip(A/"proof"/r["video"]) for k,r in items.items()}
    def frame(role,t):return clips[role][min(int(t*items[role]["fps"]),len(clips[role])-1)]
    scenes=[]
    def add(name,duration,draw):scenes.append((name,duration,draw))
    def intro(t):
        im=base("SO-101 / SMOLVLA / MUJOCO","SmolVLA on the SO-101")
        text(im,(46,133),"Fine-tuning, inference and pick-and-place evaluation",25,MUTED)
        hero=frame("success",min(t,items["success"]["simulation_duration_s"])).crop((0,64,1152,448))
        im.paste(hero.resize((1152,384)),(64,198))
        text(im,(64,612),"3 RGB cameras",25,GREEN,True);text(im,(450,612),"40k-step checkpoint",25,GREEN,True)
        text(im,(900,612),"6 GB GPU inference",25,GREEN,True)
        text(im,(64,661),"SIMULATION ONLY  /  Full results, failed trials and source attribution included",16,MUTED)
        return im
    add("Overview",5,intro)
    def success(t):
        im=base("ROLLOUT","A complete grasp, lift and placement")
        im.paste(frame("success",t),(64,159))
        text(im,(64,628),f"Seed {items['success']['seed']}  |  Source-fit materials  |  Frozen 40k policy",23,GREEN,True)
        text(im,(64,669),"1x simulated time, sampled at 9.8 fps. Rerendered saved physics states; not a live inference capture.",15,MUTED)
        return im
    add("Full successful rollout",items["success"]["simulation_duration_s"],success)
    def paired(t):
        im=base("CONTACT MATERIAL COMPARISON","Same starting positions. Different contact materials.")
        for role,x,label,color in (("paired_baseline",48,"BASELINE / FAILED",BLUE),("success",664,"CANDIDATE / SUCCESS",GREEN)):
            raw=frame(role,t)
            f=raw.crop((0,64,384,448)).resize((472,472))
            im.paste(f,(x+48,155));text(im,(x,121),label,21,color,True)
            overhead=raw.crop((768,64,1152,448)).resize((160,160))
            im.paste(overhead,(x+350,165));d=ImageDraw.Draw(im)
            d.rectangle((x+350,165,x+509,184),fill=BG);text(im,(x+355,166),"OVERHEAD",12,WHITE,True)
            d.rectangle((x+349,164,x+510,326),outline=color,width=2)
            if t>items[role]["simulation_duration_s"]:text(im,(x+14,590),"Complete - final frame held",17,WHITE,True)
        text(im,(48,645),f"Seed {items['success']['seed']}. This example does not establish aggregate improvement.",20,MUTED)
        text(im,(48,682),"Playback follows simulated time. The shorter successful rollout holds its final frame.",14,MUTED)
        return im
    add("Matched illustrative pair",items["paired_baseline"]["simulation_duration_s"],paired)
    def failure(t):
        im=base("PLACEMENT FAILURE","Reaching the bin is not the completed task")
        im.paste(frame("placement_failure",t),(64,159))
        text(im,(64,628),f"Seed {items['placement_failure']['seed']}  |  Lifted and reached bin XY  |  Placement failed",22,"#ffb09c",True)
        text(im,(64,669),"Full raw clips and every per-seed outcome are included in the evidence folder.",17,MUTED)
        return im
    add("Failed placement",items["placement_failure"]["simulation_duration_s"],failure)
    def results(t):
        im=base("PAIRED EVALUATION","Placement success on 50 paired starts")
        chart=Image.open(A/"success_comparison.png").convert("RGB");chart=fit(chart,(1184,500));im.paste(chart,((W-chart.width)//2,137))
        text(im,(48,648),"200 policy rollouts across 2 materials x 2 rendering settings",22,GREEN,True)
        text(im,(48,684),"Main chart: original 4-sample rendering. 42% observed candidate success; a reliable gain is not established.",14,MUTED)
        return im
    add("Paired results",9,results)
    def offline(t):
        im=base("CHECKPOINT EVALUATION","Held-out action error across saved checkpoints")
        chart=fit(Image.open(A/"offline_mae.png").convert("RGB"),(1184,465));im.paste(chart,((W-chart.width)//2,142))
        text(im,(48,621),"Training: Threadripper PRO / RTX 6000 Ada 48 GB",22,GREEN,True)
        text(im,(48,670),"Inference: RTX 3060 6 GB. Action prediction error and task success are separate metrics.",16,MUTED)
        return im
    add("Archived offline evaluation",8,offline)
    def engineering(t):
        im=base("CONTROL LOOP","Observation, prediction and joint control")
        labels=[("OBSERVE","3 RGB views\n6 joint angles"),("PREDICT","Frozen SmolVLA\n50 joint targets"),("EXECUTE","25 targets\nMuJoCo contacts"),("EVALUATE","Stage metrics\nStrict placement")]
        d=ImageDraw.Draw(im)
        for i,(tag,body) in enumerate(labels):
            x=48+i*305;d.rounded_rectangle((x,176,x+265,350),18,fill="#15263b",outline="#31465d")
            text(im,(x+20,198),tag,18,GREEN,True)
            for j,line in enumerate(body.split("\n")):text(im,(x+20,245+j*36),line,23)
            if i<3:text(im,(x+275,240),">",28,MUTED,True)
        text(im,(48,408),"Named joint mapping  /  Contact diagnostics  /  Reproducible run metadata",23,WHITE,True)
        text(im,(48,465),"Found a zero-tolerance contact-metric defect; exactly replayed 100 archived traces.",21,MUTED)
        text(im,(48,511),"Audited anti-aliasing sensitivity; kept measurement repair separate from better actions.",21,MUTED)
        text(im,(48,584),"Next research direction: perception-aware recovery and verified simulator fidelity.",22,GREEN,True)
        text(im,(48,665),"Credits: Hugging Face SmolVLA/LeRobot  |  gpudad dataset  |  Rovo Labs scene replica",15,MUTED)
        return im
    add("System and findings",8,engineering)
    def close(t):
        im=base("RESULTS AND SOURCE","SmolVLA on the SO-101")
        text(im,(48,176),"Recorded rollouts, metrics and source code.",30,GREEN,True)
        for j,line in enumerate(("Full success and failure clips.","Training and evaluation report.","Per-seed metrics, source hashes and evaluation limits.")):
            text(im,(48,267+j*62),line,26)
        text(im,(48,519),"Python  /  PyTorch  /  LeRobot  /  MuJoCo  /  Gymnasium",23,MUTED)
        text(im,(48,620),"Simulation study. The original simulator and physical-robot performance are not claimed.",19,MUTED)
        return im
    add("Evidence and credits",6,close)
    output=av.open(str(P/"demo.mp4"),"w");stream=output.add_stream("libx264",rate=FPS);stream.width=W;stream.height=H;stream.pix_fmt="yuv420p";stream.options={"crf":"20","preset":"fast"}
    timeline=[];offset=0
    preview=[]
    for name,duration,draw in scenes:
        frames=round(duration*FPS);timeline.append({"section":name,"start_s":offset,"duration_s":frames/FPS})
        still=draw(min(duration/2,2));preview.append(still.copy())
        for i in range(frames):
            im=draw(i/FPS)
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.array(im),format="rgb24")):output.mux(packet)
        offset+=frames/FPS;print("Composed",name,flush=True)
    for packet in stream.encode():output.mux(packet)
    output.close()
    sheet=Image.new("RGB",(1280,360),BG)
    for i,im in enumerate(preview):sheet.paste(im.resize((320,180)),((i%4)*320,(i//4)*180))
    sheet.save(A/"demo_contact_sheet.jpg",quality=95)
    preview[0].save(A/"demo_poster.jpg",quality=95)
    gif=[]
    for i in range(24):
        im=frame("success",i/24*items["success"]["simulation_duration_s"]).resize((768,299));gif.append(im)
    gif[0].save(A/"demo_preview.gif",save_all=True,append_images=gif[1:],duration=180,loop=0,optimize=True)
    (P/"demo_timeline.json").write_text(json.dumps({"duration_s":offset,"fps":FPS,"audio":"None; captions carry the explanation","sections":timeline,
        "evidence_method":manifest["method"],"limits":"Selected illustrative clips; all 50 outcomes/profile retained; no physical-robot validation or real-time hardware claim"},indent=2))
    print("Reel duration",offset,flush=True)


if __name__=="__main__":main()
