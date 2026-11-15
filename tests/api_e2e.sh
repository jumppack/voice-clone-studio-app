#!/bin/bash
# API end-to-end: consent, room check, 3 takes, generate (timed previews + target length), export
# WAV + MP3 at each bitrate (the export endpoint returns the file; it is saved where we say).
# Run against a server started with the SAME data folder, e.g.:
#   VCS_DATA=$PWD/data_test PORT=7862 NO_BROWSER=1 ./run.sh &
#   VCS_DATA=$PWD/data_test PORT=7862 tests/api_e2e.sh
set -e
B=http://127.0.0.1:${PORT:-7861}; F=tests/fixtures; D=${VCS_DATA:-data}; X=$D/test-exports
mkdir -p "$X"
PY=.venv/bin/python
curl -sf -X POST $B/api/settings -F consent=true >/dev/null
curl -sf -X POST $B/api/settings -F watermark_ack=true >/dev/null
$PY -c "import numpy as np,soundfile as sf;sf.write('$D/room.wav',np.random.default_rng(1).normal(0,10**(-62/20),96000),48000,subtype='PCM_16')"
curl -sf -X POST $B/api/room-check -F file=@$D/room.wav; echo
for i in 1 2 3; do curl -sf -X POST $B/api/takes/$i -F file=@$F/daniel_take${i}_room.wav | $PY -c "import json,sys;d=json.load(sys.stdin);print('take',d['step'],d['analysis']['score'],d['analysis']['verdict'])"; done
curl -sf $B/api/profile | $PY -c "import json,sys;d=json.load(sys.stdin);print('profile ready',d['ready'],'voice',d['voice_take'])"

run() {  # run <curl -F args...>  -> prints summary, sets J
  J=$(curl -sf -X POST $B/api/generate "$@" | $PY -c "import json,sys;print(json.load(sys.stdin)['id'])")
  while :; do S=$(curl -sf $B/api/jobs/$J); ST=$(echo "$S" | $PY -c "import json,sys;print(json.load(sys.stdin)['state'])"); [ "$ST" = done ] || [ "$ST" = error ] && break; sleep 1; done
  echo "$S" | $PY -c "import json,sys;d=json.load(sys.stdin);r=d.get('result',{});print(f\"{d['state']} target={d.get('target')} -> {r.get('seconds')} s, {len(r.get('chunks',[]))} parts, unspoken={r.get('unspoken')}, sim={r.get('similarity')}, cps={r.get('cps')}, {d['elapsed']} s\", d.get('error') or '')"
}
LONG="The garden behind our old house was never tidy, but it was always alive. Every spring my grandmother planted beans along the fence and tomatoes by the back steps. She talked to the plants while she worked, as if they were neighbours. By July the beans had climbed higher than my head. We picked them in the evening when the air finally cooled down. Some nights we ate them raw, straight from the vine, before they ever reached the kitchen. I still think of that garden whenever I smell fresh soil after rain."

echo "--- previews at 5 / 15 / 30 s"
for t in 5 15 30; do run -F preview=true -F target=$t -F 'params={"seed":7}'; done
P=$J
echo "--- long text: whole, then target 10 s and 20 s"
run -F "text=$LONG" -F 'params={"seed":7}'
for t in 10 20; do run -F "text=$LONG" -F target=$t -F 'params={"seed":7}'; done
echo "--- bad target is refused"
curl -s -o /dev/null -w "target=45 -> HTTP %{http_code}\n" -X POST $B/api/generate -F preview=true -F target=45

echo "--- export (file comes back in the response)"
curl -sf -X POST $B/api/export -F output_id=$P -F fmt=wav -F filename=test-wav -o "$X/test-wav.wav" -D - | grep -i "content-disposition"
for k in 128 192 256 320; do curl -sf -X POST $B/api/export -F output_id=$P -F fmt=mp3 -F bitrate=$k -F filename=test-$k -o "$X/test-$k.mp3"; done
ffprobe -v error -show_entries stream=codec_name,sample_rate,sample_fmt,bits_per_sample:format=duration -of compact $D/outputs/$P.wav
ffprobe -v error -show_entries stream=codec_name,sample_rate,sample_fmt,bits_per_sample:format=duration -of compact "$X/test-wav.wav"
for k in 128 192 256 320; do ffprobe -v error -show_entries stream=codec_name,sample_rate,bit_rate:format=duration -of compact "$X/test-$k.mp3"; done
ffmpeg -v error -i "$X/test-320.mp3" -f null - && echo "mp3 decodes cleanly"
ls "$D"/cache | grep -c "^export-" | xargs echo "temp export files left behind:"
