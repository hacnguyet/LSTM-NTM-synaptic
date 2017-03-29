var video = document.querySelector('video');
var range = document.querySelector('input');
var audioCtx = new AudioContext();
var analyser = audioCtx.createAnalyser();

// getUserMedia block - grab stream
// put it into a MediaStreamAudioSourceNode
// also output the visuals into a video element

if (navigator.mediaDevices) {
    console.log('getUserMedia supported.');
    navigator.mediaDevices.getUserMedia ({audio: true, video: true})
    .then(function(stream) {
        // Create a MediaStreamAudioSourceNode
        // Feed the HTMLMediaElement into it
        var source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);

        analyser.fftSize = 2048;
        var bufferLength = analyser.frequencyBinCount;
        var dataArray = new Uint8Array(bufferLength);
        analyser.getByteTimeDomainData(dataArray);

        // Get a canvas defined with ID "oscilloscope"
        var canvas = document.getElementById("oscilloscope");
        var canvasCtx = canvas.getContext("2d");

        
        // draw an oscilloscope of the current audio source

        function draw() {

          drawVisual = requestAnimationFrame(draw);

          analyser.getByteTimeDomainData(dataArray);
          //console.log(dataArray);

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

        // Create a biquadfilter
        var biquadFilter = audioCtx.createBiquadFilter();
        //biquadFilter.frequency.value = 1000;
        //biquadFilter.gain.value = range.value;

        // connect the AudioBufferSourceNode to the gainNode
        // and the gainNode to the destination, so we can play the
        // music and adjust the volume using the mouse cursor
        source.connect(biquadFilter);
        console.log(biquadFilter.frequency.value, biquadFilter.gain.value);
        biquadFilter.connect(audioCtx.destination);

        // Get new mouse pointer coordinates when mouse is moved
        // then set new gain value

        range.oninput = function() {
            //biquadFilter.gain.value = range.value;
        }
    })
    .catch(function(err) {
        console.log('The following gUM error occured: ' + err);
    });
} else {
   console.log('getUserMedia not supported on your browser!');
}

