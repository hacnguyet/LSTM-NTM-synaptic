// define audio api variables
var sampleRate = 3000;
var sampleLength = 0.2;
var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
var offlineCtx = new OfflineAudioContext(1,sampleRate*sampleLength,sampleRate);
var source = offlineCtx.createBufferSource();
var song = audioCtx.createBufferSource();
var audioData, readBuffer, audioBuffer, audioBuffer_test, audioBuffer_json;

//define synaptic variables
var Neuron = synaptic.Neuron,
    Layer = synaptic.Layer,
    Network = synaptic.Network,
    Trainer = synaptic.Trainer,
    Architect = synaptic.Architect;

var LSTM = new Architect.LSTM(5,20,20,20,1);
var iterations = 1000;
var rate = 0.01;
var trial,res;

// use FileReader to load an audio track, and
// decodeAudioData to decode it and stick it in a buffer.
function openFile(event) {
  var input = event.target;
  var reader = new FileReader();

  reader.onload = function() {
    audioData = reader.result;
    audioCtx.decodeAudioData(audioData, function(buffer) {
      readBuffer = buffer;
      source = offlineCtx.createBufferSource();
      source.buffer = readBuffer;
      source.connect(offlineCtx.destination);
      source.start();
      offlineCtx.startRendering().then(function(renderedBuffer) {
        console.log('Rendering completed successfully');
        audioBuffer = renderedBuffer;
        for (var channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
          var nowBuffering = audioBuffer.getChannelData(channel);
          for (var i = 0; i < audioBuffer.length; i++) {
            //nowBuffering[i] = nowBuffering[i] * 2;
          }
        }
      }).catch(function(err) {
          console.log('Rendering failed: ' + err);
          // Note: The promise should reject when startRendering is called a second time on an OfflineAudioContext
      });  
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
  audioBuffer_test = audioCtx.createBuffer(1, sampleRate * sampleLength, sampleRate);
  var nowBuffering = audioBuffer_test.getChannelData(0);
  var testBuffering = audioBuffer.getChannelData(0);
  nowBuffering[0] = testBuffering[0];
  nowBuffering[1] = testBuffering[1];
  nowBuffering[2] = testBuffering[2];
  nowBuffering[3] = testBuffering[3];
  nowBuffering[4] = testBuffering[4];
  for (var i = 0; i < audioBuffer_test.length - 5; i++) {
    res = LSTM.activate([(nowBuffering[i]+1)/2,(nowBuffering[i+1]+1)/2,(nowBuffering[i+2]+1)/2,(nowBuffering[i+3]+1)/2,(nowBuffering[i+4]+1)/2]);
    nowBuffering[i+5] = res[0] * 2 - 1;
  }  
  play(audioBuffer_test, 'oscilloscope_test');
}

//Put buffer to source then play
function play_json() {
  console.log("Creating audioBuffer from json");
  audioBuffer_json = audioCtx.createBuffer(1, sampleRate * sampleLength, sampleRate);
  var nowBuffering = audioBuffer_json.getChannelData(0);
  var parsed = JSON.parse($('#textarea').val());
  for(var x in nowBuffering){
    nowBuffering[x] = parsed[x];
  }
  //log audioBuffer info
  console.log(audioBuffer_json.sampleRate, audioBuffer_json.length, audioBuffer_json.duration, audioBuffer_json.numberOfChannels);

  song = audioCtx.createBufferSource();
  song.buffer = audioBuffer_json;
  song.connect(audioCtx.destination);
  var dataArray = audioBuffer_json.getChannelData(0);
  $('#textarea').val(JSON.stringify(dataArray));

  // Get a canvas defined with ID "oscilloscope"
  var canvas = document.getElementById('oscilloscope_test');
  var canvasCtx = canvas.getContext("2d");

  
  // draw an oscilloscope of the current audio source
  canvasCtx.fillStyle = 'rgb(200, 200, 200)';
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

  canvasCtx.beginPath();

  var sliceWidth = canvas.width * 1.0 / dataArray.length;
  var x = 0;

  for (var i = 0; i < dataArray.length; i++) {

    var v = (dataArray[i] + 1) / 2;
    var y = v * canvas.height;

    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }

    x += sliceWidth;
  }

  canvasCtx.lineTo(canvas.width, canvas.height / 2);
  canvasCtx.stroke();

  //play song
  song.start(0);
}

//Put buffer to source then play
function play(buffer, canvas_id) {
  //log audioBuffer info
  console.log(buffer.sampleRate, buffer.length, buffer.duration, buffer.numberOfChannels);

  song = audioCtx.createBufferSource();
  song.buffer = buffer;
  song.connect(audioCtx.destination);
  var dataArray = buffer.getChannelData(0);
  $('#textarea').val(JSON.stringify(dataArray));

  // Get a canvas defined with ID "oscilloscope"
  var canvas = document.getElementById(canvas_id);
  var canvasCtx = canvas.getContext("2d");

  
  // draw an oscilloscope of the current audio source
  canvasCtx.fillStyle = 'rgb(200, 200, 200)';
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

  canvasCtx.beginPath();

  var sliceWidth = canvas.width * 1.0 / dataArray.length;
  var x = 0;

  for (var i = 0; i < dataArray.length; i++) {

    var v = (dataArray[i] + 1) / 2;
    var y = v * canvas.height;

    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }

    x += sliceWidth;
  }

  canvasCtx.lineTo(canvas.width, canvas.height / 2);
  canvasCtx.stroke();

  //play song
  song.start(0);
}

function stop() {
  song.stop(0);
}