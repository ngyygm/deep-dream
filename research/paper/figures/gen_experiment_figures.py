"""Publication figures from frozen evidence; no model calls or data mutation.

All four experiments use final paper width 5.5in. Authored SVG mechanism
figures have separate exporters and must never be regenerated here.
"""
from pathlib import Path
import json
import hashlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
V = json.loads((ROOT.parent/'results/recomputed_values.json').read_text())
L = json.loads((ROOT.parent/'results/evidence_ledger.json').read_text())
BLUE, TEAL, ROSE = '#24669B', '#18888B', '#B64C72'
INK, MUTED, GRID = '#193D55', '#5C7282', '#D5E0E6'
TRACKS = ['baseline', 'skill-agent', 'pi']
LABELS = ['Direct', 'Memory agent', 'Source agent']
PATH_COLORS = [BLUE, TEAL, ROSE]
MARKERS = ['o', 's', '^']
DOMAINS = ['AR', 'TTL', 'LRU', 'SF']
mpl.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':7, 'axes.labelsize':7,
    'axes.titlesize':7.5, 'axes.titleweight':'bold', 'axes.titlecolor':INK,
    'xtick.labelsize':6.3, 'ytick.labelsize':6.3, 'legend.fontsize':6.2,
    'text.color':INK, 'axes.labelcolor':MUTED, 'xtick.color':MUTED,
    'ytick.color':MUTED, 'axes.edgecolor':GRID, 'axes.linewidth':.65,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'savefig.bbox':None, 'savefig.dpi':300, 'figure.dpi':150,
})


def style(ax, axis='x'):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=.45, alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=.5, pad=2)


def title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left', pad=8)


def export(fig, stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=fig.bbox
    outside=[]
    sizes=[]
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip(): continue
        sizes.append(t.get_fontsize())
        box=t.get_window_extent(renderer)
        # Matplotlib places a colourbar label in a rotated glyph box that can
        # extend a few pixels past the renderer bbox although it is not clipped
        # in the saved page.  Keep a generous final-size tolerance and still
        # fail on genuine panel text escaping the canvas.
        if box.x0 < -18 or box.y0 < -18 or box.x1 > bounds.x1+18 or box.y1 > bounds.y1+18:
            outside.append(t.get_text())
    assert min(sizes)>=6, (stem, min(sizes))
    assert not outside, (stem, 'outside canvas', outside)
    for ext in ['pdf','svg','png']:
        fig.savefig(ROOT/f'{stem}.{ext}', dpi=300, bbox_inches=None)
    with Image.open(ROOT/f'{stem}.png') as im:
        ImageOps.grayscale(im).save(ROOT/f'{stem}_gray.png', dpi=(300,300))
    print(f'{stem}: {fig.get_size_inches()} in; min font={min(sizes):.1f}pt; bounds PASS')
    plt.close(fig)


def heatmap(ax, values, xlabels, ylabels, fig, cbar=True):
    a=np.asarray(values)
    ny,nx=a.shape
    # Sequential, perceptually uniform and grayscale-safe colormap.
    im=ax.pcolormesh(np.arange(nx+1),np.arange(ny+1),a,
                     cmap='cividis',vmin=0,vmax=100,edgecolors='white',linewidth=.5)
    ax.set_xlim(0,nx); ax.set_ylim(ny,0)
    ax.set_xticks(np.arange(nx)+.5,xlabels)
    ax.set_yticks(np.arange(ny)+.5,ylabels)
    ax.tick_params(length=0, pad=3)
    ax.spines[:].set_visible(False)
    for i in range(ny):
        for j in range(nx):
            ax.text(j+.5,i+.5,f'{a[i,j]:.1f}',ha='center',va='center',fontsize=6.4,
                    color='white' if a[i,j]<45 else INK)
    if cbar:
        cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.035,ticks=[0,50,100])
        cb.ax.tick_params(labelsize=6,length=2,pad=1)


def ladder():
    d=V['ladder']
    fig,axes=plt.subplots(2,2,figsize=(5.5,3.7))
    fig.subplots_adjust(left=.16,right=.96,top=.9,bottom=.14,wspace=.60,hspace=.95)
    a,b,c,e=axes.flat
    title(a,'a','Overall quality')
    y=np.arange(3)[::-1]
    for j,version in enumerate(['v1','v2']):
        vals=[d[f'overall_{version}_x100'][t] for t in TRACKS]
        ci=np.array([d[f'overall_ci95_{version}_x100'][t] for t in TRACKS])
        yy=y+(.15 if j==0 else -.15)
        for idx,(v,lo,hi) in enumerate(zip(vals,ci[:,0],ci[:,1])):
            a.errorbar(v,yy[idx],xerr=[[v-lo],[hi-v]],fmt='o' if j==0 else 'D',
                       ms=3.2,color=PATH_COLORS[idx],mfc='white' if j==0 else PATH_COLORS[idx],
                       elinewidth=.75,capsize=1.8,mew=.7)
            a.text(v+2,yy[idx]+(.09 if j==0 else -.12),f'{v:.1f}',fontsize=6,va='bottom' if j==0 else 'top')
    a.set_yticks(y,LABELS); a.set_xlim(0,100); a.set_ylim(-.55,2.65)
    a.set_xticks([0,50,100]); a.set_xlabel('Overall (×100)'); style(a)
    a.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=MUTED,label='Base ($n{=}767$)',ms=3),
                      Line2D([],[],marker='D',ls='',color=MUTED,label='Align+ ($n{=}1074$)',ms=3)],
             loc='upper left',bbox_to_anchor=(-.02,1.13),frameon=False,ncol=2,columnspacing=1)
    title(b,'b','Memory capabilities')
    heatmap(b,[[d['domains_v2_x100'][t][dm] for t in TRACKS] for dm in DOMAINS],
            ['Direct','Memory','Source'],DOMAINS,fig)
    title(c,'c','Selected tasks')
    tasks=[('MCC','MCC'),('FC-MH','FC-MH'),('LME(S*)','LME-S'),('DetQA','DetectiveQA')]
    for j,(col,marker,label) in enumerate(zip(PATH_COLORS,MARKERS,LABELS)):
        vals=[d['tasks_v2_x100'][TRACKS[j]][k] for k,_ in tasks]
        c.scatter(vals,np.arange(4)[::-1]+(j-1)*.19,c=col,marker=marker,s=12,label=label,zorder=3)
    c.set_yticks(np.arange(4)[::-1],[v for _,v in tasks]);c.set_xlim(0,100);c.set_ylim(-.5,3.5)
    c.set_xticks([0,50,100]);c.set_xlabel('Task score (×100)');style(c)
    fig.legend(*c.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),
               ncol=3,frameon=False,columnspacing=1.8,handletextpad=.4)
    title(e,'d','Remaining zero scores')
    tax=V['failure_taxonomy_v2_pi']
    counts=[tax['by_domain_count'][k] for k in DOMAINS]
    assert sum(counts)==tax['zero_score_questions']
    e.barh(np.arange(4)[::-1],counts,height=.55,color=ROSE,alpha=.8)
    for yy,k,n in zip(np.arange(4)[::-1],DOMAINS,counts):
        e.text(n+1,yy,f"{n} ({tax['by_domain_percent'][k]:.1f}%)",va='center',fontsize=6)
    e.set_yticks(np.arange(4)[::-1],DOMAINS);e.set_xlim(0,140);e.set_xticks([0,50,100])
    e.set_xlabel(f"Questions ({tax['zero_score_questions']} total)");style(e)
    export(fig,'fig2_ladder')


def cost():
    fig,axes=plt.subplots(2,2,figsize=(5.5,3.7))
    fig.subplots_adjust(left=.17,right=.96,top=.90,bottom=.12,wspace=.56,hspace=.85)
    a,b,c,d=axes.flat
    title(a,'a','Answer latency')
    p95max=0
    for i in range(3):
        z=V['rung_cost_v2'][f'rung{i+1}'];y=2-i
        assert z['questions']==1074
        lo,hi=z['latency_seconds_mean'],z['latency_seconds_p95']
        p95max=max(p95max,hi)
        a.plot([lo,hi],[y,y],c=PATH_COLORS[i],lw=2,alpha=.35)
        a.scatter([lo],[y],marker='o',c=PATH_COLORS[i],s=14)
        a.scatter([hi],[y],marker='D',facecolors='white',edgecolors=PATH_COLORS[i],s=14)
        a.text(9,y-.28,f"{z['tokens_per_question']/1000:.1f}k tokens · {z['llm_calls_per_question']:.1f} calls",fontsize=6,color=MUTED)
    a.set_yticks([2,1,0],LABELS);a.set_xlim(0,1.06*p95max);a.set_ylim(-.6,2.7)
    a.set_xticks([0,150,300,450]);a.set_xlabel('Seconds / question');style(a)
    a.legend(handles=[Line2D([],[],ls='',marker='o',color=MUTED,label='Mean',ms=3),
                      Line2D([],[],ls='',marker='D',color=MUTED,mfc='white',label='p95',ms=3)],
             loc='upper left',bbox_to_anchor=(-.02,1.11),ncol=2,frameon=False)
    title(b,'b','Ingest cost')
    ec=L['four_benchmark']['engine_comparison']; groups=['mab','lme'];x=np.arange(2)
    for j,key in enumerate(['llm_calls_per_doc','tokens_per_doc']):
        ratios=[]; raw=[]
        for g in groups:
            z=ec[g]['per_doc_intersection' if g=='mab' else 'per_doc']
            base,align=z['v1'][key],z['v2'][key]
            ratios.append(100*align/base)
            scale=1000 if j else 1
            raw.append(f'{base/scale:.0f}→{align/scale:.0f}'+('k' if j else ''))
        xx=x+(j-.5)*.31
        b.bar(xx,ratios,width=.29,color=[BLUE,TEAL][j],hatch=['','///'][j],
              label=['Calls','Tokens'][j],edgecolor='white',linewidth=.4)
        for k in range(2):b.text(xx[k],ratios[k]+3,raw[k],ha='center',va='bottom',fontsize=6)
    b.axhline(100,c=MUTED,ls='--',lw=.7);b.set_ylim(0,115);b.set_yticks([0,50,100])
    b.set_xticks(x,['MAB','LME']);b.set_ylabel('Cost (% of Base)');style(b,'y')
    b.legend(loc='upper left',bbox_to_anchor=(-.02,1.11),frameon=False,ncol=2)
    title(c,'c','Evidence budget')
    z=L['depth_diagnostic'];assert z['invariant_violations']==0
    ks=np.asarray(z['ks']);rows=[z['per_k'][str(k)] for k in ks]
    payload=np.array([r['mean_evidence_payload_bytes']/1000 for r in rows])
    assert np.all(np.diff(payload)>0)
    for name,col,marker,ls,lab in [('any',BLUE,'o','-','Any'),('all',TEAL,'s','--','All')]:
        mean=np.array([r[f'recall_{name}_pct'] for r in rows])
        lo=[r[f'recall_{name}_ci_low_pct'] for r in rows];hi=[r[f'recall_{name}_ci_high_pct'] for r in rows]
        c.fill_between(payload,lo,hi,color=col,alpha=.12,lw=0)
        c.plot(payload,mean,c=col,marker=marker,ms=3,lw=1,ls=ls,label=lab)
    for xx,k,r in zip(payload,ks,rows):
        c.annotate(str(k),(xx,r['recall_any_pct']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6)
    c.set_xlabel('Evidence payload (kB)');c.set_ylabel('Recall (%)');c.set_xlim(0,4.1);c.set_ylim(0,100)
    c.set_xticks([0,1,2,3,4]);c.set_yticks([0,50,100]);style(c,'y')
    c.legend(loc='lower right',frameon=False,handlelength=1.5)
    title(d,'d','Channel replay at k=10')
    arms=[f'arm{i}_{t}' for i,t in enumerate(['source_lexical','source_semantic','source_neighbors','source_relations'],1)]
    for j,(name,col,marker,lab) in enumerate([('any',BLUE,'o','Any'),('all',TEAL,'s','All')]):
        for i,key in enumerate(arms):
            z=L['provenance_ablation_x7']['arms'][key]['per_k']['10'];v=z[f'recall_{name}_pct']
            lo,hi=z[f'recall_{name}_ci_low_pct'],z[f'recall_{name}_ci_high_pct']
            d.errorbar(v,3-i+(j-.5)*.24,xerr=[[v-lo],[hi-v]],fmt=marker,color=col,
                       ms=3,elinewidth=.6,capsize=1.6,label=lab if i==0 else None)
    d.set_yticks([3,2,1,0],['Lexical','+ Semantic','+ Neighbors','+ Relations'])
    d.set_xlim(0,100);d.set_ylim(-.6,3.6);d.set_xticks([0,50,100]);d.set_xlabel('Recall (%)');style(d)
    d.legend(loc='upper left',bbox_to_anchor=(-.02,1.10),ncol=2,frameon=False)
    export(fig,'fig3_cost')


def alignment():
    s=V['same_question_alignment']; assert s['n']==614
    tracks=s['tracks'];labels=LABELS+['Full context']
    cols=PATH_COLORS+[MUTED];marks=MARKERS+['D']
    fig,(a,b)=plt.subplots(1,2,figsize=(5.5,1.95),gridspec_kw={'width_ratios':[1,1.03]})
    fig.subplots_adjust(left=.19,right=.96,top=.84,bottom=.27,wspace=.32)
    title(a,'a','Same-question Overall')
    for i,t in enumerate(tracks):
        v=s['overall_x100'][t];lo,hi=s['overall_ci95_x100'][t];y=3-i
        a.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt=marks[i],color=cols[i],
                   ms=4,capsize=2,elinewidth=.8)
        a.text(99,y,f'{v:.1f}',ha='right',va='center',fontsize=6.2)
    a.set_yticks(range(3,-1,-1),labels);a.set_xlim(0,104);a.set_xticks([0,50,100])
    a.set_ylim(-.5,3.5);a.set_xlabel('Overall (×100)');style(a)
    title(b,'b','Capability scores')
    heatmap(b,[[s['domains_x100'][t][dm] for dm in DOMAINS] for t in tracks],DOMAINS,['']*4,fig)
    fig.text(.19,.05,'Same rows in both panels · Kimi-K3 · 614 common questions',fontsize=6.2,color=MUTED)
    export(fig,'figA_alignment')


def outcomes():
    ec=L['four_benchmark']['engine_comparison']
    fig,axes=plt.subplots(1,3,figsize=(5.5,1.75))
    fig.subplots_adjust(left=.10,right=.97,top=.73,bottom=.29,wspace=.48)
    a,b,c=axes
    for ax,field,label in [(a,'calls','Calls / document'),(b,'duplicates','Same-name families')]:
        for i,g in enumerate(['mab','lme']):
            z=ec[g]
            if field=='calls':
                r=z['per_doc_intersection' if g=='mab' else 'per_doc']
                pair=[r[v]['llm_calls_per_doc'] for v in ['v1','v2']]
            else: pair=[z['library_dbs'][v]['dup_name_rows'] for v in ['v1','v2']]
            ax.plot(pair,[1-i]*2,c=GRID,lw=2.2,zorder=1)
            for j,v in enumerate(pair):
                ax.scatter(v,1-i,marker=['o','D'][j],c=[MUTED,TEAL][j],s=17,zorder=3)
                fmt=f'{v:.1f}' if field=='calls' else str(v)
                ax.annotate(fmt,(v,1-i),xytext=(0,7 if j==0 else -12),textcoords='offset points',ha='center',fontsize=6)
        ax.set_yticks([1,0],['MAB','LME']);ax.set_ylim(-.5,1.6)
        ax.set_xlim(0,690 if field=='calls' else 550);ax.set_xticks([0,300,600] if field=='calls' else [0,250,500])
        ax.set_xlabel(label,fontsize=6.5);style(ax)
    title(a,'a','Ingest calls');title(b,'b','Duplicate families');title(c,'c','Score change')
    for i,key in enumerate(['ttl_mcc','fc_mh']):
        z=V['consolidation_paired'][key];v=100*z['delta'];lo,hi=np.array(z['delta_ci95'])*100
        assert z['n']==100
        c.errorbar(v,1-i,xerr=[[v-lo],[hi-v]],fmt='D',c=ROSE,ms=4,capsize=2,elinewidth=1)
        c.annotate(f'{100*z["v1"]:.0f}→{100*z["v2"]:.0f}',(v,1-i),xytext=(0,8),textcoords='offset points',ha='center',fontsize=6)
    c.axvline(0,c=MUTED,ls='--',lw=.6);c.set_xlim(-15,20);c.set_xticks([-10,0,10,20])
    c.set_ylim(-.5,1.6);c.set_yticks([1,0],['MCC','FC-MH']);c.set_xlabel('Align+ − Base (pp)',fontsize=6.5);style(c)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',c=MUTED,label='Base',ms=3.5),
                        Line2D([],[],marker='D',ls='',c=TEAL,label='Align+',ms=3.5)],
               loc='upper center',bbox_to_anchor=(.35,.99),frameon=False,ncol=2)
    export(fig,'figB_alignment_outcomes')


if __name__=='__main__':
    ladder();cost();alignment();outcomes()
