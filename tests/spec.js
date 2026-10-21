module.exports=async(page)=>{
  await page.waitForFunction(()=>document.querySelector('#train-prog')&&document.querySelector('#train-prog').textContent.includes('trained'),{timeout:120000});
  await page.type('#rev','This product is absolutely amazing!!! Best purchase ever!!! Everyone should buy this right now!!! You will not regret it!!! Five stars!!!');
  await page.click('#analyze');
  await page.waitForSelector('#result:not(.hidden)',{timeout:20000});
  const v1=await page.$eval('#verdict',e=>e.textContent);
  if(!/fake/i.test(v1))throw new Error('fake not detected: '+v1);
  await page.evaluate(()=>{document.getElementById('rev').value='';document.getElementById('fake-num').textContent='—'});
  await page.type('#rev','Used this daily for two months. Battery easily lasts a full workday and the build feels solid. The app is a bit clunky and syncing took two tries, but overall good value for the price.');
  await page.click('#analyze');
  await page.waitForFunction(()=>document.querySelector('#fake-num').textContent!=='—',{timeout:20000});
  const v2=await page.$eval('#verdict',e=>e.textContent);
  if(!/genuine/i.test(v2))throw new Error('genuine not detected: '+v2);
  const sigs=await page.$eval('#signals',e=>e.children.length);
  if(sigs<7)throw new Error('signal breakdown missing: '+sigs);
  const acc=await page.$eval('#m-acc',e=>e.textContent);
  return `fake+genuine detected, 7 signals shown, CV acc=${acc}`;
};
