/* fake-review-detector — logistic regression over bag-of-words + handcrafted
   deception signals, trained in the browser. Ships with pre-trained weights. */
(function(){
'use strict';
const $=id=>document.getElementById(id);
const STOP=new Set('a an the is are was were i you he she it we they me him her us them my your his their our this that these those and or but if then so of to in on at for with by from as be been have has had do does did will would can could should not no'.split(' '));
const NEGATORS=new Set(['not','never','no','hardly','barely',"didn't","doesn't","don't","wasn't","isn't","couldn't","won't","cant","can't","without"]);
const HYPE=new Set('amazing perfect best incredible flawless love obsessed masterpiece phenomenal revolutionary sensational outstanding exceptional unbelievable greatest awesome fantastic brilliant superb stunning thrilled magic miracle'.split(' '));
const COMPLAINT=new Set('broke broken defect defective disappointed disappointing terrible awful worst useless poor bad filthy dirty slow rude damaged stale cheap flimsy boring frustrating failed waste'.split(' '));
function tokenize(t){return t.toLowerCase().replace(/[^a-z0-9'\s]/g,' ').split(/\s+/).filter(w=>w.length>1)}
function stem(w){if(w.length<=3)return w;
  for(const s of['ingly','edly','ing','ed','ly','es','s'])if(w.endsWith(s)&&w.length-s.length>=3)return w.slice(0,w.length-s.length);
  return w}
function feats(t){
  const ws=tokenize(t),out=[];
  for(let i=0;i<ws.length;i++){
    if(NEGATORS.has(ws[i])&&i+1<ws.length){out.push('not_'+stem(ws[i+1]));i++;continue}
    if(!STOP.has(ws[i]))out.push(stem(ws[i]));
  }
  return out;
}
const SIGNALS=[
 {k:'exclam',name:'Exclamation density',hint:'!!! bursts are a classic fake-review tell'},
 {k:'caps',name:'ALL-CAPS ratio',hint:'SHOUTING words correlate with planted hype'},
 {k:'hype',name:'Superlative density',hint:'generic praise words vs concrete detail'},
 {k:'firstp',name:'First-person ratio',hint:'fakes often underuse or overuse "I/my"'},
 {k:'specif',name:'Specificity (numbers, units)',hint:'real reviews mention sizes, times, quantities'},
 {k:'sentlen',name:'Average sentence length',hint:'fakes tend to run very short or very long'},
 {k:'repet',name:'Word repetition',hint:'farms reuse the same words over and over'}];
function handFeats(text){
  const raw=text.split(/\s+/).filter(Boolean);
  const ws=tokenize(text);
  const n=Math.max(ws.length,1);
  const sents=Math.max((text.match(/[.!?]+/g)||[]).length,1);
  const uniq=new Set(ws);
  return {
    exclam:(text.match(/!/g)||[]).length/n,
    caps:raw.filter(w=>w.length>2&&w===w.toUpperCase()&&/[A-Z]/.test(w)).length/n,
    hype:ws.filter(w=>HYPE.has(stem(w))).length/n,
    firstp:ws.filter(w=>/^(i|me|my|mine|myself|we|our|us)$/.test(w)).length/n,
    specif:ws.filter(w=>/\d/.test(w)||/^(inch|inches|cm|mm|kg|g|km|m|mile|miles|litre|liter|ml|oz|lb|lbs|hour|hours|minute|minutes|day|days|week|weeks|month|months|year|years|size)$/.test(w)).length/n,
    sentlen:Math.min(ws.length/sents,60)/60,
    repet:1-uniq.size/n
  };
}
const N_HAND=SIGNALS.length;
let VOCAB=[],model=null,trained=false,VOCAB_SIZE=0,metrics=null,hMean=null,hStd=null;
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function seededShuffle(arr,seed){const r=mulberry32(seed);for(let i=arr.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[arr[i],arr[j]]=[arr[j],arr[i]]}return arr}
const allTexts=()=>[...GENUINE_REVIEWS,...FAKE_REVIEWS];
function buildVocab(){
  const freq={};
  allTexts().forEach(t=>new Set(feats(t)).forEach(f=>freq[f]=(freq[f]||0)+1));
  VOCAB=Object.keys(freq).filter(f=>freq[f]>=2);
  VOCAB_SIZE=VOCAB.length+2;
  const rows=allTexts().map(handFeats);
  hMean=SIGNALS.map(s=>rows.reduce((a,r)=>a+r[s.k],0)/rows.length);
  hStd=SIGNALS.map(s=>{const m=hMean[SIGNALS.indexOf(s)];const v=rows.reduce((a,r)=>a+(r[s.k]-m)**2,0)/rows.length;return Math.sqrt(v)||1});
}
function vectorize(text){
  const v=new Array(VOCAB_SIZE+N_HAND).fill(0);
  feats(text).forEach(f=>{const i=VOCAB.indexOf(f);if(i>=0)v[i+2]=1});
  const h=handFeats(text);
  SIGNALS.forEach((s,j)=>{v[VOCAB_SIZE+j]=(h[s.k]-hMean[j])/hStd[j]});
  return {v,words:tokenize(text)};
}
function createModel(){
  const m=tf.sequential();
  m.add(tf.layers.dense({units:1,inputShape:[VOCAB_SIZE+N_HAND],activation:'sigmoid',
    kernelRegularizer:tf.regularizers.l2({l2:0.01})}));
  m.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  return m;
}
const EPOCHS=300,BATCH=32,FOLDS=5;
const corpus=()=>[...GENUINE_REVIEWS.map(t=>({t,y:0})),...FAKE_REVIEWS.map(t=>({t,y:1}))]; // y=1 means fake
async function fitModel(rows){
  const m=createModel();
  const x=tf.tensor2d(rows.map(d=>vectorize(d.t).v));
  const y=tf.tensor1d(rows.map(d=>d.y));
  await m.fit(x,y,{epochs:EPOCHS,batchSize:BATCH,shuffle:true,verbose:0});
  x.dispose();y.dispose();
  return m;
}
async function predictProbs(m,rows){
  const x=tf.tensor2d(rows.map(d=>vectorize(d.t).v));
  const pv=await m.predict(x).data();
  x.dispose();
  return Array.from(pv);
}
async function crossValidate(prog){
  const data=corpus();
  seededShuffle(data,42);
  const oof=new Array(data.length);
  for(let k=0;k<FOLDS;k++){
    const te=data.filter((_,i)=>i%FOLDS===k),tr=data.filter((_,i)=>i%FOLDS!==k);
    const m=await fitModel(tr);
    const pv=await predictProbs(m,te);
    te.forEach((d,i)=>oof[data.indexOf(d)]=pv[i]);
    m.dispose();
    if(prog)prog(`Cross-validating… fold ${k+1}/${FOLDS}`);
  }
  let tp=0,tn=0,fp=0,fn=0;
  oof.forEach((p,i)=>{const c=p>=0.5?1:0,y=data[i].y;if(c===1&&y===1)tp++;else if(c===0&&y===0)tn++;else if(c===1)fp++;else fn++});
  const prec=tp/(tp+fp||1),rec=tp/(tp+fn||1);
  return{acc:(tp+tn)/data.length,prec,rec,f1:2*prec*rec/(prec+rec||1),n:data.length,tp,tn,fp,fn};
}
async function train(prog){
  metrics=await crossValidate(prog);
  model=await fitModel(corpus());
  trained=true;
  renderMetrics();
}
function renderMetrics(){
  if(!metrics)return;
  $('m-acc').textContent=(metrics.acc*100).toFixed(1)+'%';
  $('m-f1').textContent=(metrics.f1*100).toFixed(1)+'%';
  $('m-n').textContent=metrics.n;
  $('m-conf').innerHTML=`<tr><th></th><th>Predicted fake</th><th>Predicted genuine</th></tr>
  <tr><th>Actually fake</th><td class="cm-ok">${metrics.tp}</td><td class="cm-bad">${metrics.fn}</td></tr>
  <tr><th>Actually genuine</th><td class="cm-bad">${metrics.fp}</td><td class="cm-ok">${metrics.tn}</td></tr>`;
}
async function score(text){const {v}=vectorize(text);const t=tf.tensor2d([v]);const p=(await model.predict(t).data())[0];t.dispose();return p}
async function handWeights(){
  const w=model.getWeights()[0];
  const vals=await w.data();
  return SIGNALS.map((s,j)=>({...s,raw:vals[VOCAB_SIZE+j]}));
}
function verdictFor(p){return p>=0.65?['Likely fake','bad']:p>=0.4?['Suspicious','warn']:['Likely genuine','ok']}
async function analyze(){
  if(!trained){pkToast('Model is still loading, one moment…');return}
  const text=$('rev').value.trim();
  if(!text){pkToast('Paste a review first');return}
  const p=await score(text);
  const h=handFeats(text);
  const hw=await handWeights();
  document.getElementById('result').classList.remove('hidden');
  const[lab,cls]=verdictFor(p);
  $('verdict').innerHTML=`<span class="pill ${cls}">${lab}</span>`;
  $('fake-num').textContent=(p*100).toFixed(1)+'%';
  $('fake-meter').style.width=(p*100)+'%';
  $('fake-meter').style.background=p>=0.65?'var(--bad)':p>=0.4?'#d97706':'var(--ok)';
  const stars=parseInt($('stars').value||'0',10);
  let mismatch='';
  if(stars){
    const hypeN=tokenize(text).filter(w=>HYPE.has(stem(w))).length;
    const negN=tokenize(text).filter(w=>COMPLAINT.has(stem(w))).length;
    if(stars<=2&&hypeN>1&&negN===0)mismatch=` You picked ${stars}★ but the text is all praise — a rating/text mismatch, common in manipulated listings.`;
    else if(stars>=4&&negN>hypeN+1)mismatch=` You picked ${stars}★ but the text is mostly complaints — a rating/text mismatch worth a second look.`;
  }
  $('fake-note').textContent='Fake-likelihood score — under 40% looks genuine, 40–65% is suspicious, over 65% looks fake.'+mismatch;
  $('signals').innerHTML=hw.map((s,j)=>{
    const contrib=s.raw*(h[s.k]-hMean[j])/hStd[j];
    const dir=contrib>0.05?'bad':contrib<-0.05?'ok':'warn';
    const pct=Math.min(Math.abs(contrib)*60,100);
    const val=h[s.k];
    const disp=s.k==='sentlen'?(val*60).toFixed(1)+' words':(val*100).toFixed(1)+'%';
    return `<div class="sig"><div class="sig-head"><span>${s.name}</span><span class="mono">${disp}</span></div>
    <div class="sig-bar"><span class="sig-fill ${dir}" style="width:${Math.max(pct,3)}%"></span></div>
    <div class="sig-note">${contrib>0.05?'Pushes toward fake':contrib<-0.05?'Pushes toward genuine':'Neutral'} — ${s.hint}</div></div>`;
  }).join('');
  const {words}=vectorize(text);
  const contribs=[];
  for(let i=0;i<words.length;i++){
    if(STOP.has(words[i]))continue;
    const w2=words.slice();w2.splice(i,1);
    const {v}=vectorize(w2.join(' '));
    const t=tf.tensor2d([v]);
    const q=(await model.predict(t).data())[0];
    t.dispose();
    contribs.push({w:words[i],d:p-q});
  }
  const esc=text.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  const fakeW=new Map(contribs.filter(c=>c.d>0.005).map(c=>[c.w,c.d]));
  const genW=new Map(contribs.filter(c=>c.d<-0.005).map(c=>[c.w,-c.d]));
  $('highlighted').innerHTML=esc.split(/\b/).map(part=>{
    const lw=part.toLowerCase();
    if(fakeW.has(lw))return `<mark class="hl-bad" title="+${fakeW.get(lw).toFixed(2)} fake">${part}</mark>`;
    if(genW.has(lw))return `<mark class="hl-ok" title="-${genW.get(lw).toFixed(2)} fake">${part}</mark>`;
    return part;
  }).join('');
  const topF=contribs.filter(c=>c.d>0.005).sort((a,b)=>b.d-a.d).slice(0,6);
  const topG=contribs.filter(c=>c.d<-0.005).sort((a,b)=>a.d-b.d).slice(0,6);
  $('fake-words').innerHTML=topF.map(c=>`<span class="pill bad">${c.w}</span>`).join(' ')||'';
  $('gen-words').innerHTML=topG.map(c=>`<span class="pill ok">${c.w}</span>`).join(' ')||'';
}
function jaccard(a,b){const A=new Set(a),B=new Set(b);let inter=0;A.forEach(x=>{if(B.has(x))inter++});return inter/(A.size+B.size-inter||1)}
async function batch(){
  if(!trained){pkToast('Model is still loading, one moment…');return}
  const raw=$('batch-in').value.trim();if(!raw){pkToast('Paste reviews, one per line');return}
  const lines=raw.split('\n').map(l=>l.trim()).filter(Boolean);
  const rows=[];
  for(const l of lines){rows.push({l,p:await score(l),toks:tokenize(l)})}
  const dup=new Set();
  for(let i=0;i<rows.length;i++)for(let j=i+1;j<rows.length;j++)
    if(jaccard(rows[i].toks,rows[j].toks)>=0.5){dup.add(i);dup.add(j)}
  $('batch-out').classList.remove('hidden');
  $('batch-tbl').innerHTML=rows.map((r,i)=>{
    const[lab,cls]=verdictFor(r.p);
    const d=dup.has(i)?' <span class="pill warn">near-duplicate</span>':'';
    return `<tr><td>${i+1}</td><td>${r.l.length>70?r.l.slice(0,70)+'…':r.l}</td><td><span class="pill ${cls}">${lab}</span>${d}</td><td class="mono">${(r.p*100).toFixed(0)}%</td></tr>`}).join('');
  const nF=rows.filter(r=>r.p>=0.65).length,nG=rows.filter(r=>r.p<0.4).length;
  $('batch-sum').innerHTML=`<span class="pill bad">${nF} likely fake</span> <span class="pill warn">${rows.length-nF-nG} suspicious</span> <span class="pill ok">${nG} likely genuine</span>${dup.size?` <span class="pill warn">${dup.size} near-duplicates</span>`:''}`;
  $('batch-dl').classList.remove('hidden');
  $('batch-dl').onclick=()=>pkDownload('fake-review-results.csv',pkCsv([['review','verdict','fake_score','near_duplicate'],...rows.map((r,i)=>[r.l,verdictFor(r.p)[0],r.p.toFixed(3),dup.has(i)?'yes':'no'])]),'text/csv');
}
function readyUI(source){
  trained=true;
  $('stat-vocab').textContent=VOCAB.length.toLocaleString();
  $('stat-texts').textContent=allTexts().length.toLocaleString();
  $('stat-params').textContent=model.countParams().toLocaleString();
  $('stat-type').textContent='LogReg';
  $('train-prog').innerHTML='<span class="pill ok">Model ready</span> '+source;
  renderMetrics();
}
function base64ToBuf(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer}
function bufToBase64(buf){const u=new Uint8Array(buf);let s='';for(let i=0;i<u.length;i+=8192)s+=String.fromCharCode.apply(null,u.subarray(i,i+8192));return btoa(s)}
async function exportRun(){
  buildVocab();
  metrics=await crossValidate(t=>{$('train-prog').textContent=t});
  model=await fitModel(corpus());
  const art=await model.save(tf.io.withSaveHandler(async a=>({modelTopology:a.modelTopology,weightSpecs:a.weightSpecs,weightData:bufToBase64(a.weightData)})));
  art.metrics=metrics;art.vocab=VOCAB;
  window.EXPORT=JSON.stringify(art);
  $('train-prog').textContent='EXPORT READY';
}
async function boot(){
  buildVocab();
  if(location.search.includes('export'))return exportRun();
  try{
    const r=await fetch('weights.json');
    if(r.ok){
      const art=await r.json();
      art.weightData=base64ToBuf(art.weightData);
      model=await tf.loadLayersModel(tf.io.fromMemory(art));
      metrics=art.metrics||null;
      readyUI('Pre-trained weights loaded — runs fully offline.');
      return;
    }
  }catch(e){console.warn('weights load failed, training live',e)}
  train(t=>{$('train-prog').textContent=t}).then(()=>readyUI('Trained live in your browser — cross-validated metrics below.')).catch(e=>{$('train-prog').textContent='Model failed: '+e.message;console.error(e)});
}
document.addEventListener('DOMContentLoaded',()=>{
  $('analyze').onclick=analyze;
  $('batch-btn').onclick=batch;
  $('sample').onclick=()=>{
    const fake=FAKE_REVIEWS[Math.floor(Math.random()*FAKE_REVIEWS.length)];
    const gen=GENUINE_REVIEWS[Math.floor(Math.random()*GENUINE_REVIEWS.length)];
    $('rev').value=Math.random()<0.5?fake:gen;analyze()};
  const rb=$('retrain');
  if(rb)rb.onclick=async()=>{
    if(rb.disabled)return;rb.disabled=true;
    $('train-prog').textContent='Retraining in your browser…';
    try{await train(t=>{$('train-prog').textContent=t});readyUI('Retrained live in your browser — cross-validated metrics below.')}catch(e){$('train-prog').textContent='Training failed: '+e.message}
    rb.disabled=false;
  };
  boot();
});
})();
