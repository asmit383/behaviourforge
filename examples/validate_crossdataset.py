"""Cross-dataset validation — the model is fitted on Aalto and tested against KeyRecs.

Every KS check elsewhere in this library compares generated output against the corpus it was
fitted to, which is circular: it proves the sampler reproduces its input, not that the output
resembles a human. This is the external check. KeyRecs (99 people, CC BY 4.0) was collected by
different researchers, from different people, under a different protocol, and no part of it
touches the fitting.

The result splits cleanly, and the split is the point:

  SHAPE generalises.       Hold CV 0.37 against our 0.36; latency CV 0.90 against our 0.96.
                           The distributional claims — log-normal, right-skewed, these widths
                           — hold on a corpus the model has never seen.

  CORRELATION does not.    KeyRecs shows a lag-1 autocorrelation of +0.001 where Aalto shows
                           +0.056, and the per-person hold/latency correlation flips sign
                           (-0.163 against +0.16). Neither is a stable human constant; both
                           are properties of the TASK. Aalto is transcription — copy a shown
                           sentence, continuous flow, so tempo drifts. A fixed-phrase protocol
                           resets tempo every trial and the drift disappears.

The practical consequence is in keystroke.TEMPO_SIGMA_SCALE, which is calibrated to Aalto and
is therefore right for continuous prose and probably too high for short form fields.

    python examples/validate_crossdataset.py
"""
import os, sys, zipfile, json, math, statistics as st, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from behaviourforge import forge
from behaviourforge.keystroke import Keyboard

def cv(v): return st.pstdev(v)/st.mean(v)
def acf1(xs):
    lg=[math.log(x) for x in xs if x>0]
    if len(lg)<10: return None
    m=sum(lg)/len(lg); d=sum((x-m)**2 for x in lg)
    return sum((lg[i]-m)*(lg[i+1]-m) for i in range(len(lg)-1))/d if d else None
def pear(a,b):
    n=len(a); ma,mb=sum(a)/n,sum(b)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da and db else 0

# --- KeyRecs (theirs, never used in fitting) ---
z=zipfile.ZipFile(os.environ.get('KEYRECS_ZIP','/tmp/ah.zip'))
d=json.loads(z.read([x for x in z.namelist() if x.endswith('data/keystrokes.json')][0]))
kr=[]
for p in d['people']:
    s=p['s'] if isinstance(p,dict) else p
    h=[];l=[]
    for i in range(0,len(s)-4,5):
        _,_,_,hold,lat=s[i:i+5]
        if 10<=hold<=1000: h.append(hold)
        if 0<lat<=2000: l.append(lat)
    if len(h)>=50 and len(l)>=50: kr.append((h,l))

# --- ours: fitted on Aalto only ---
SENTS=["please confirm your address before we ship the order",
       "i will send the documents over later this afternoon",
       "let me know whether the meeting time still works"]
ours=[]
for s in range(99):
    p=forge(s); kb=Keyboard(p.motor, random.Random(s))
    h=[];l=[]
    for i in range(12):
        for k in kb.plan(SENTS[i%3], first_field=True, typos=False):
            h.append(k.dwell_ms)
            if k.flight_ms<=2000: l.append(k.flight_ms)
    ours.append((h,l))

def block(name, data):
    H=[x for h,_ in data for x in h]; L=[x for _,l in data for x in l]
    hm=[st.median(h) for h,_ in data]; lm=[st.median(l) for _,l in data]
    a=[acf1(l) for _,l in data]; a=[x for x in a if x is not None]
    return dict(n=len(data), hold=st.median(H), hcv=cv(H), lat=st.median(L), lcv=cv(L),
                acf=st.mean(a), corr=pear(lm,hm))

k=block("KeyRecs",kr); o=block("ours",ours)
print(f"{'':34}{'KeyRecs':>10}{'ours':>10}{'Aalto':>10}")
print("="*64)
print(f"{'people':<34}{k['n']:>10}{o['n']:>10}{'6000':>10}")
print(f"{'SHAPE — generalises?':<34}")
print(f"{'  hold CV':<34}{k['hcv']:>10.2f}{o['hcv']:>10.2f}{'0.34':>10}")
print(f"{'  latency CV':<34}{k['lcv']:>10.2f}{o['lcv']:>10.2f}{'1.00':>10}")
print(f"{'LOCATION — task dependent':<34}")
print(f"{'  hold median ms':<34}{k['hold']:>10.0f}{o['hold']:>10.0f}{'104':>10}")
print(f"{'  latency median ms':<34}{k['lat']:>10.0f}{o['lat']:>10.0f}{'152':>10}")
print(f"{'CORRELATION — generalises?':<34}")
print(f"{'  lag-1 acf of latency':<34}{k['acf']:>+10.4f}{o['acf']:>+10.4f}{'+0.056':>10}")
print(f"{'  per-person corr(hold,latency)':<34}{k['corr']:>+10.3f}{o['corr']:>+10.3f}{'+0.16':>10}")
