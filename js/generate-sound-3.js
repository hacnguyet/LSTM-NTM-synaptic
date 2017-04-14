// define audio api variables
var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
var analyser = audioCtx.createAnalyser();
var source,audioData,audioBuffer,audioBuffer_test;

//define synaptic variables
var Neuron = synaptic.Neuron,
    Layer = synaptic.Layer,
    Network = synaptic.Network,
    Trainer = synaptic.Trainer,
    Architect = synaptic.Architect;

var LSTM = new Architect.LSTM(5,20,20,1);
var iterations = 1000;
var rate = 0.001;
var trial,res;

// use FileReader to load an audio track, and
// decodeAudioData to decode it and stick it in a buffer.
function openFile(event) {
  var input = event.target;
  var reader = new FileReader();

  reader.onload = function() {
    audioData = reader.result;
    audioCtx.decodeAudioData(audioData, function(buffer) {
      audioBuffer = buffer;
      for (var channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        var nowBuffering = audioBuffer.getChannelData(channel);
        for (var i = 0; i < audioBuffer.length; i++) {
          nowBuffering[i] = nowBuffering[i];
        }
      }
    },function(e){ console.log("Error with decoding audio data" + e.err); });
    console.log("Loading complete!");
  }
  reader.readAsArrayBuffer(input.files[0]);
};

//Train LSTM with buffer data
function train(){
  var start = Date.now();
  console.log("Start train");
  for (var channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
    var nowBuffering = audioBuffer.getChannelData(channel);
    for(var j = 0; j < iterations; j++){
      for (var i = 0; i < audioBuffer.length - 5; i++) {
        res = LSTM.activate([(nowBuffering[i]+1)/2,(nowBuffering[i+1]+1)/2,(nowBuffering[i+2]+1)/2,(nowBuffering[i+3]+1)/2,(nowBuffering[i+4]+1)/2]);
        if(res[0] != (nowBuffering[i+5]+1)/2){
          LSTM.propagate(rate,[(nowBuffering[i+5]+1)/2]);
        }
      }
      console.log("Finish "+(j+1)+" trains in "+(Date.now()-start)+"ms");
    }  
  }
  console.log("Finish all trains in "+(Date.now()-start)+"ms");
}

function test(){
  console.log("Creating test audioBuffer");
  audioBuffer_test = audioCtx.createBuffer(1, 1 * audioBuffer.sampleRate, audioBuffer.sampleRate);
  for (var channel = 0; channel < audioBuffer_test.numberOfChannels; channel++) {
    var nowBuffering = audioBuffer_test.getChannelData(channel);
    nowBuffering[0] = -0.27557632327079773;
    nowBuffering[1] = -0.40871551632881165;
    nowBuffering[2] = -0.3406055271625519;
    nowBuffering[3] = -0.297277569770813;
    nowBuffering[4] = -0.21418309211730957;
    for (var i = 0; i < audioBuffer_test.length - 5; i++) {
      res = LSTM.activate([(nowBuffering[i]+1)/2,(nowBuffering[i+1]+1)/2,(nowBuffering[i+2]+1)/2,(nowBuffering[i+3]+1)/2,(nowBuffering[i+4]+1)/2]);
      nowBuffering[i+5] = res[0] * 2 - 1;
    }  
    // for (var i = 0; i < audioBuffer_test.length-1; i++) {
    //   nowBuffering[i] *= 10;
    // }  
  }
  play(audioBuffer_test);
}

//Put buffer to source then play
function play(buffer) {
  //log audioBuffer info
  console.log(buffer.sampleRate, buffer.length, buffer.duration, buffer.numberOfChannels);
  console.log(buffer.getChannelData(0));

  source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.connect(analyser);

  analyser.fftSize = 2048;
  var bufferLength = analyser.frequencyBinCount;
  var dataArray = new Uint8Array(bufferLength);
  //analyser.getByteTimeDomainData(dataArray);

  // Get a canvas defined with ID "oscilloscope"
  var canvas = document.getElementById("oscilloscope");
  var canvasCtx = canvas.getContext("2d");

  
  // draw an oscilloscope of the current audio source
  function draw() {
    drawVisual = requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'rgb(200, 200, 200)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

    canvasCtx.beginPath();

    var sliceWidth = canvas.width * 1.0 / bufferLength;
    var x = 0;

    for (var i = 0; i < bufferLength; i++) {

      var v = dataArray[i] / 128.0;
      var y = v * canvas.height / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  };

  draw();
  source.start(0);
}

function stop() {
  source.stop(0);
}