"""Editable, vector-only paper Figures 1 and 5. Run on the paper host.

Figure numbers are resolved from main.aux: motivation=1, architecture=5.
The architecture example is illustrative, never an experimental trace.
The document/Episode/family projection follows the frontend; individual
observations and versions are exposed in the selected-concept detail.
No claim of measured entropy or automatic semantic entailment is encoded.
"""
from pathlib import Path
from html import escape
import argparse
import xml.etree.ElementTree as ET

BLUE = '#24669B'
TEAL = '#18888B'
ROSE = '#B64C72'
INK = '#193D55'
MUTED = '#5C7282'
LIGHT = '#D5E0E6'
PALEBLUE = '#EAF2F9'
PALETEAL = '#E7F4F2'
PALEROSE = '#F8EBF0'
WHITE = '#FFFFFF'
FONT = 'Arial, Helvetica, sans-serif'


class SVG:
    def __init__(self, width, height, title, description):
        self.w, self.h = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>', '<defs>']
        for name, col in [('blue', BLUE), ('teal', TEAL), ('rose', ROSE), ('ink', INK), ('muted', MUTED)]:
            self.parts.append(f'<marker id="{name}" markerUnits="userSpaceOnUse" viewBox="0 0 12 10" refX="10" refY="5" markerWidth="12" markerHeight="10" orient="auto"><path d="M0 0 L11 5 L0 10 Z" fill="{col}"/></marker>')
        self.parts += ['</defs>', f'<rect width="{width}" height="{height}" fill="white"/>']
        self.text_records = []

    def add(self, s): self.parts.append(s)
    def group(self, name): self.add(f'<g id="{name}">')
    def end(self): self.add('</g>')
    def box(self, x, y, w, h, fill=WHITE, stroke='none', radius=15, sw=1.8):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def ellipse(self, x, y, rx, ry, fill):
        self.add(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}"/>')
    def circle(self, x, y, r, fill=WHITE, stroke=BLUE, sw=2):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def path(self, d, color=BLUE, sw=2.2, dash=None, arrow=None, fill='none'):
        extras = (f' stroke-dasharray="{dash}"' if dash else '') + (f' marker-end="url(#{arrow})"' if arrow else '')
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{extras}/>')
    def line(self, x1, y1, x2, y2, color=BLUE, sw=2.2, dash=None, arrow=None):
        self.path(f'M{x1} {y1} L{x2} {y2}', color, sw, dash, arrow)
    def text(self, x, y, label, size=21, color=INK, weight=400, anchor='start', limit=None):
        self.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(label)}</text>')
        self.text_records.append((x, y, label, size, weight, anchor, limit))
    def lines(self, x, y, labels, size=21, color=INK, weight=400, anchor='start', gap=None, limit=None):
        for i, label in enumerate(labels): self.text(x, y+i*(gap or size*1.3), label, size, color, weight, anchor, limit)
    def badge(self, x, y, label, color=BLUE, fill=PALEBLUE, w=135):
        self.box(x, y, w, 32, fill, radius=16)
        self.text(x+w/2, y+23, label, 18, color, 700, 'middle', w-16)
    def document(self, x, y, scale=1, color=BLUE, fill=WHITE, highlight=False):
        self.add(f'<g transform="translate({x} {y}) scale({scale})">')
        self.path('M3 3 H58 L78 23 V99 H3 Z', color, 2.6, fill=fill)
        self.path('M58 3 V23 H78', color, 2.6)
        if highlight: self.box(14, 56, 53, 13, PALEROSE, radius=2)
        for yy, xx in [(36,62),(49,64),(62,64),(76,53)]: self.line(16,yy,xx,yy,color,2.6)
        self.end()
    def corpus(self, x, y, scale=1, color=BLUE):
        self.document(x+22*scale,y,scale,color,PALEBLUE)
        self.document(x+11*scale,y+11*scale,scale,color,WHITE)
        self.document(x,y+22*scale,scale,color,WHITE)
    def graph(self, x, y, scale=1):
        self.add(f'<g transform="translate({x} {y}) scale({scale})">')
        for a,b in [((0,43),(37,6)),((0,43),(42,80)),((37,6),(81,43)),((42,80),(81,43)),((81,43),(121,5)),((81,43),(132,78))]:
            self.line(*a,*b,TEAL,3)
        for xx,yy,r,c,f in [(0,43,13,BLUE,PALEBLUE),(37,6,15,TEAL,PALETEAL),(42,80,12,BLUE,PALEBLUE),(81,43,18,TEAL,PALETEAL),(121,5,13,BLUE,PALEBLUE),(132,78,13,TEAL,PALETEAL)]:
            self.circle(xx,yy,r,f,c,2.6)
        self.end()
    def check(self, x,y,scale=1,color=TEAL):
        self.circle(x,y,24*scale,PALETEAL,color,2.3)
        self.path(f'M{x-11*scale} {y} l{8*scale} {8*scale} l{16*scale} {-19*scale}',color,3)
    def search(self,x,y,color=BLUE):
        self.circle(x,y,17,WHITE,color,2.8)
        self.line(x+12,y+12,x+26,y+26,color,3.4)
    def write(self, root, name):
        payload = '\n'.join(self.parts+['</svg>'])
        xml = ET.fromstring(payload)
        assert not xml.findall('.//{http://www.w3.org/2000/svg}image')
        # Match the font metrics used by CairoSVG; detect clipped text before export.
        import cairocffi as cairo
        ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
        for x,y,label,size,weight,anchor,limit in self.text_records:
            ctx.select_font_face('Arial', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if weight>=600 else cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(size)
            bx,by,bw,bh,advance,_ = ctx.text_extents(label)
            shift = 0 if anchor=='start' else advance/2 if anchor=='middle' else advance
            assert x+bx-shift>=-1 and x+bx+bw-shift<=self.w+1, (name,'horizontal clipping',label)
            assert y+by>=-1 and y+by+bh<=self.h+1, (name,'vertical clipping',label)
            if limit: assert advance<=limit, (name,'label too wide',label,advance,limit)
        dest = root/(name+'.svg')
        dest.write_text(payload,encoding='utf-8')
        import cairosvg
        # 468 pt is the ICLR text width; no raster image is embedded in the PDF.
        cairosvg.svg2pdf(bytestring=payload.encode(),write_to=str(root/(name+'.pdf')),output_width=624,output_height=624*self.h/self.w)
        cairosvg.svg2png(bytestring=payload.encode(),write_to=str(root/(name+'.png')),output_width=self.w,output_height=self.h)
        print(f'{name}: SVG/PDF/PNG, {self.w}x{self.h}, {len(self.text_records)} editable labels; geometry passed')


def motivation():
    s=SVG(1440,470,'Memory addressing and adaptive evidence reading',
          'Figure 1. Fixed top-k retrieval and query-time progressive disclosure. The graph addresses source spans; the agent can return to search after a read. The provenance gate checks read identifiers, not answer correctness.')
    s.group('fixed-read')
    s.box(10,10,450,442,'#F3F6F8',radius=25)
    s.text(35,52,'a',27,BLUE,700)
    s.text(72,52,'Fixed read',29,INK,700)
    s.text(35,86,'One retrieval sets the answer context',21,MUTED,limit=405)
    s.ellipse(98,278,61,13,'#E1E8ED')
    s.corpus(52,140,.85,BLUE)
    s.line(151,210,197,210,BLUE,2.8,arrow='blue')
    s.box(213,157,82,105,WHITE,BLUE,8,2.4)
    for yy,width in [(177,51),(193,43),(209,49),(225,37),(241,46)]:
        s.line(229,yy,229+width,yy,'#8EAFC8',4)
    s.line(309,210,347,210,BLUE,2.8,arrow='blue')
    s.path('M365 175 H419 Q431 175 431 187 V225 Q431 237 419 237 H385 L365 254 V237 Q353 237 353 225 V187 Q353 175 365 175',BLUE,2.4,fill=WHITE)
    s.line(366,194,416,194,BLUE,3)
    s.line(366,211,406,211,BLUE,3)
    s.text(98,318,'Documents',21,INK,700,'middle')
    s.text(254,318,'Top-k spans',21,INK,700,'middle')
    s.text(393,318,'Answer',21,INK,700,'middle')
    s.line(40,352,430,352,LIGHT,1.3)
    s.text(40,390,'Fixed evidence budget',23,BLUE,700)
    s.text(40,423,'No feedback to select another read',20,MUTED)
    s.end()

    s.group('adaptive-read')
    s.box(478,10,952,442,'#F1F8F8',radius=25)
    s.text(503,52,'b',27,TEAL,700)
    s.text(540,52,'Deep-Dream',29,INK,700)
    s.badge(1073,29,'Agent-controlled read',TEAL, '#DDEEEE',328)
    s.text(503,86,'Concepts locate evidence; the agent decides what to open next',21,MUTED)
    s.ellipse(612,281,99,13,'#DCECEC')
    s.graph(543,160,.95)
    s.line(701,210,775,210,TEAL,2.8,arrow='teal')
    s.text(738,188,'locate',18,TEAL,400,'middle')
    s.document(793,153,.96,ROSE,WHITE,True)
    s.document(881,153,.96,ROSE,WHITE,True)
    s.line(973,210,1033,210,TEAL,2.8,arrow='teal')
    s.check(1081,210,1.55)
    s.line(1129,210,1193,210,TEAL,2.8,arrow='teal')
    s.document(1234,147,1.02,BLUE)
    s.check(1320,242,.9)
    s.text(609,314,'Concept + relation graph',21,INK,700,'middle')
    s.text(873,314,'Source spans',21,INK,700,'middle')
    s.text(1082,314,'Read-ID check',21,INK,700,'middle')
    s.text(1280,314,'Cited answer',21,INK,700,'middle')
    s.text(1082,339,'provenance only',17,MUTED,400,'middle')
    # The feedback loop changes the next read; it does not fabricate source facts.
    s.path('M873 333 V375 Q873 387 860 387 H621 Q608 387 608 373 V341',TEAL,2.5,dash='7 6',arrow='teal')
    s.box(670,367,173,36,'#F1F8F8',radius=8)
    s.text(756,391,'Need more evidence?',18,TEAL,700,'middle')
    s.text(1113,397,'Search  /  expand  /  read  /  stop',20,TEAL,700,'middle')
    s.text(1113,426,'Read budget follows the question',19,MUTED,400,'middle')
    s.end()
    return s


def architecture():
    s=SVG(1600,820,'A document-first temporal concept graph',
          'Figure 5. Illustrative document/Episode/entity-family graph and selected-concept history. Three documents share the Trip concept through provenance links. A Paris plan is updated to Rome and later confirmed. Each observation remains linked to its own original source; processing time is the sole state axis. Relations connect concepts, and uncertain families may be redirected after additional evidence. This is not a measured benchmark case.')
    s.text(30,40,'a',27,BLUE,700)
    s.text(65,40,'Across documents: shared concepts',29,INK,700)
    s.text(1013,40,'b',27,TEAL,700)
    s.text(1048,40,'Within a concept: history + source',29,INK,700)
    s.text(30,72,'Documents retain their Episodes; mentions meet at a stable family.',21,MUTED)
    s.text(1013,72,'Selecting Trip opens its linked observations.',21,MUTED)
    s.box(14,95,955,602,'#F5F8FB',radius=26)
    s.box(993,95,591,602,'#F0F7F6',radius=26)
    # Provenance and relation edges first so labels and nodes remain clear.
    s.group('document-episode-links')
    for cx in [170,474,778]: s.line(cx,224,cx,278,BLUE,2.2,arrow='blue')
    s.text(31,264,'contains',18,MUTED)
    s.end()
    s.group('cross-document-mentions')
    s.path('M170 364 C170 432 350 394 432 465',TEAL,2.3,dash='6 6',arrow='teal')
    s.line(474,364,474,448,TEAL,2.3,dash='6 6',arrow='teal')
    s.path('M778 364 C778 419 589 397 518 465',TEAL,2.3,dash='6 6',arrow='teal')
    s.text(477,408,'mentions',18,TEAL,400,'middle')
    s.end()
    s.group('relation-edges')
    s.line(209,505,416,505,ROSE,2.4,arrow='rose')
    s.text(309,491,'plans',20,ROSE,400,'middle')
    s.path('M522 481 C590 441 650 458 712 458',ROSE,2.2,dash='7 5',arrow='rose')
    s.text(614,441,'destination · earlier',18,ROSE,400,'middle')
    s.path('M523 527 C592 565 647 576 712 576',ROSE,2.6,arrow='rose')
    s.text(614,607,'destination · later',18,ROSE,400,'middle')
    s.end()
    s.group('documents')
    docs=[(170,'notes.md','E1 · Plan'),(474,'update.md','E2 · Change'),(778,'receipt.md','E3 · Confirm')]
    for cx,name,episode in docs:
        s.ellipse(cx,210,91,12,'#E0EAF1')
        s.document(cx-35,112,.72,BLUE)
        s.text(cx,215,name,24,INK,700,'middle')
        s.box(cx-112,287,224,77,WHITE,'#ADC8DE',15)
        s.text(cx,318,episode,23,BLUE,700,'middle')
        s.text(cx,345,{'notes.md':'Paris planned','update.md':'Rome replaces Paris','receipt.md':'Rome confirmed'}[name],20,MUTED,400,'middle',208)
    s.end()
    s.group('concept-families')
    s.circle(162,505,46,PALETEAL,TEAL,2.4)
    s.text(162,513,'Alex',26,INK,700,'middle')
    s.circle(474,505,59,'none','#A1D0C9',1.6)
    s.circle(474,505,51,'#CEE9E3',TEAL,3)
    s.text(474,513,'Trip',30,INK,700,'middle')
    s.text(474,588,'shared family',21,TEAL,700,'middle')
    s.text(474,614,'f_trip',18,MUTED,400,'middle')
    for cx,cy,name in [(760,458,'Paris'),(760,576,'Rome')]:
        s.circle(cx,cy,46,PALETEAL,TEAL,2.2)
        s.text(cx,cy+8,name,25,INK,700,'middle')
    s.text(60,662,'Cross-document links arise through shared families and relations.',21,INK,limit=858)
    s.end()
    # A selected family expands to observations, not a second invented graph hierarchy.
    s.group('selected-concept')
    s.circle(1036,135,17,'#CEE9E3',TEAL,2)
    s.text(1068,143,'Trip',26,INK,700)
    s.text(1150,142,'stable identity: f_trip',20,MUTED)
    s.badge(1398,117,'Selected',TEAL,'#D8EEEA',152)
    s.text(1056,192,'Processing / observation time',21,TEAL,700)
    s.line(1037,220,1037,478,TEAL,2.4,arrow='teal')
    s.text(1502,192,'version chain',18,MUTED,400,'end')
    rows=[(216,'1','Paris','notes.md · E1',PALEBLUE,BLUE),
          (311,'2','Rome','update.md · E2',PALEROSE,ROSE),
          (406,'3','Rome','receipt.md · E3',PALETEAL,TEAL)]
    for y,num,value,src,fill,color in rows:
        s.circle(1037,y+34,7,color,color,1)
        s.box(1064,y,486,70,WHITE,'#BBD6D1',12,1.4)
        s.badge(1079,y+19,'rev. '+num,color,fill,90)
        s.text(1190,y+43,value,24,color,700)
        s.line(1261,y+35,1293,y+35,MUTED,1.6,arrow='muted')
        s.text(1306,y+41,src,19,MUTED,limit=230)
    s.text(1064,512,'Each revision resolves to its own source.',21,TEAL,700,limit=478)
    s.box(1020,538,534,132,WHITE,'#DFC5CF',16,1.5)
    s.document(1037,557,.60,ROSE,WHITE,True)
    s.text(1102,568,'Original evidence · update.md / E2',20,ROSE,700,limit=425)
    s.lines(1102,598,['“Alex changed the destination','from Paris to Rome.”'],23,INK,gap=29,limit=425)
    s.text(1102,652,'Earlier evidence is still addressable.',19,MUTED)
    s.end()
    # Compact revision mechanism, visually distinct from the frontend projection.
    s.group('progressive-alignment')
    s.box(14,717,1570,86,'#F6F8F9',radius=18)
    s.text(35,750,'Progressive alignment',22,INK,700)
    s.text(35,780,'Identity can be revised.',19,MUTED)
    s.circle(337,746,14,PALETEAL,TEAL,1.8)
    s.circle(365,774,14,PALETEAL,TEAL,1.8)
    s.text(398,749,'Uncertain match',20,TEAL,700)
    s.text(398,775,'keep families separate',18,MUTED)
    s.line(613,760,681,760,TEAL,2.4,arrow='teal')
    s.document(710,734,.43,BLUE)
    s.text(766,750,'More evidence',20,BLUE,700)
    s.text(766,776,'content + relation context',18,MUTED)
    s.line(1002,760,1070,760,TEAL,2.4,arrow='teal')
    s.circle(1115,760,23,'#CEE9E3',TEAL,2)
    s.text(1155,750,'Redirect compatible families',21,TEAL,700,limit=402)
    s.text(1155,776,'preserve observations and source links',18,MUTED,limit=402)
    s.end()
    return s


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only',choices=['motivation','architecture','all'],default='all')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    if args.only in ('motivation','all'): motivation().write(args.out,'fig0_motivation')
    if args.only in ('architecture','all'): architecture().write(args.out,'fig1_architecture')


if __name__=='__main__': main()
