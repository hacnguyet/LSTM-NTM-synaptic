var audioCtx = new (window.AudioContext || window.webkitAudioContext)();

// Stereo
var channels = 2;
// Create an empty two second stereo buffer at the
// sample rate of the AudioContext
var frameCount = audioCtx.sampleRate * 2.0;

var myAudioBuffer = audioCtx.createBuffer(channels, frameCount, audioCtx.sampleRate);

function play() {
  // Fill the buffer with white noise;
  //just random values between -1.0 and 1.0
  for (var channel = 0; channel < channels; channel++) {
   // This gives us the actual ArrayBuffer that contains the data
   var nowBuffering = myAudioBuffer.getChannelData(channel);
   for (var i = 0; i < frameCount; i++) {
     // Math.random() is in [0; 1.0]
     // audio needs to be in [-1.0; 1.0]
     nowBuffering[i] = Math.random() * 2 - 1;
   }
  }

  // Get an AudioBufferSourceNode.
  // This is the AudioNode to use when we want to play an AudioBuffer
  var source = audioCtx.createBufferSource();
  // set the buffer in the AudioBufferSourceNode
  source.buffer = myAudioBuffer;
  // connect the AudioBufferSourceNode to the
  // destination so we can hear the sound
  source.connect(audioCtx.destination);
  // start the source playing
  source.start();
}