// Raw PCM capture. Posts level stats ~every 21 ms, and copies of the samples while recording.
class RawRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.buf = new Float32Array(1024);
    this.n = 0;
    this.peak = 0;
    this.sum = 0;
    this.port.onmessage = (e) => { if (e.data && "recording" in e.data) this.recording = e.data.recording; };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const v = ch[i];
      this.buf[this.n++] = v;
      const a = v < 0 ? -v : v;
      if (a > this.peak) this.peak = a;
      this.sum += v * v;
      if (this.n === this.buf.length) {
        const msg = { peak: this.peak, rms: Math.sqrt(this.sum / this.n) };
        if (this.recording) msg.samples = this.buf.slice(0);
        this.port.postMessage(msg);
        this.n = 0; this.peak = 0; this.sum = 0;
      }
    }
    return true;
  }
}
registerProcessor("raw-recorder", RawRecorder);
