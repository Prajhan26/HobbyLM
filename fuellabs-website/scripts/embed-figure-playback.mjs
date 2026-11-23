// Playback-only adaptations of the locked figure sources for the local preview.
export function withAutomaticPlayback(html, id) {
  if (id === 'B2-F3') {
    const adapter = `
var autoWanted=!matchMedia('(prefers-reduced-motion: reduce)').matches,autoVisible=false,autoView='both';
var originalPlayIcon=bp.innerHTML;
var pauseIcon='<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M4.5 3.25v9.5M11.5 3.25v9.5" stroke="currentColor" stroke-width="1.5"/></svg>';
var autoButton=bp.cloneNode(true);bp.replaceWith(autoButton);bp=autoButton;
function autoRender(){var on=autoWanted&&autoVisible&&!document.hidden;bp.innerHTML=on?pauseIcon:originalPlayIcon;bp.setAttribute('aria-label',on?'Pause explanation':'Play explanation');bp.setAttribute('aria-pressed',String(on))}
function autoTick(){clearTimeout(timer);if(!autoWanted||!autoVisible||document.hidden)return;timer=setTimeout(function(){autoView=autoView==='a'?'b':autoView==='b'?'both':'a';show(autoView);autoTick()},autoView==='both'?2200:1700)}
function autoSync(){clearTimeout(timer);autoRender();autoTick()}
bp.addEventListener('click',function(){autoWanted=!autoWanted;if(autoWanted&&autoView==='both'){autoView='a';show(autoView)}autoSync()});
vb.forEach(function(button){button.addEventListener('click',function(){autoWanted=false;autoView=button.dataset.v;autoSync()})});
new IntersectionObserver(function(entries){var was=autoVisible;autoVisible=entries[0].isIntersecting;if(autoVisible&&!was&&autoWanted&&autoView==='both'){autoView='a';show(autoView)}autoSync()},{threshold:.05}).observe(fig);
document.addEventListener('visibilitychange',autoSync);
matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',function(event){autoWanted=!event.matches;if(event.matches){autoView='both';show(autoView)}autoSync()});
autoRender();
`;
    // The adapter owns playback; the original manual view buttons are retained.
    html = html.replace(/if\(!rm&&'IntersectionObserver' in window\)\{[^\n]+\n/, '');
    return html.replace(/\n\}\)\(\);\s*\n<\/script>/, `\n${adapter}\n})();\n</script>`);
  }
  if (!html.includes('var step=') || !html.includes("id=\"b-play\"")) return html;
  const last = id === 'B1-I1' ? '3' : 'LAST';
  const adapter = `
var autoWanted=!matchMedia('(prefers-reduced-motion: reduce)').matches,autoVisible=false,autoStarted=false;
schedule=function(){clearTimeout(timer);if(!playing||!autoWanted||!autoVisible||document.hidden)return;timer=setTimeout(function(){step=step>=${last}?0:step+1;render();schedule()},step>=${last}?2200:HOLD[step])};
function autoSync(){clearTimeout(timer);playing=autoWanted&&autoVisible&&!document.hidden;if(playing&&!autoStarted){autoStarted=true;started=true;step=0}render();schedule()}
bp.addEventListener('click',function(){autoWanted=!playing},true);
bp.addEventListener('click',function(){if(autoWanted)autoStarted=true;autoSync()});
[bv,bn].forEach(function(button){button.addEventListener('click',function(){autoWanted=false},true)});
new IntersectionObserver(function(entries){autoVisible=entries[0].isIntersecting;autoSync()},{threshold:.05}).observe(fig);
document.addEventListener('visibilitychange',autoSync);
matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',function(event){autoWanted=!event.matches;if(event.matches){step=${last};playing=false}autoSync()});
`;
  // Avoid competing with the fan figure's existing one-time start observer.
  html = html.replace(/^var auto=false;[^\n]*\n/m, '');
  return html.replace(/\n\}\)\(\);\s*\n<\/script>/, `\n${adapter}\n})();\n</script>`);
}
