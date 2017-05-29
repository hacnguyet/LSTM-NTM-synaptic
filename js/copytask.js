var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

var LSTM = new Architect.LSTM(8,20,20,20,20,20,8);
var iterations = 1000;
var rate = .1;
var input = [];
var output = [];
var prediction = [];

function train(){
	console.log('Start train');
	train_length = document.getElementById("train-length").value;
	iterations = document.getElementById("iterations").value;
	if(train_length == '')
		train_length = 2; 
	if(iterations == '')
		iterations = 1000;   
	var start = Date.now();
	var correct = 0;
	for(var i = 0; i < iterations; i++){
		input = [];
		var train_length_random = Math.floor((Math.random() * train_length) + 1);
		for(var j = 0; j < train_length_random; j++){
			input.push([Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random())]);
		}
		for(var j in input){
			output = LSTM.activate(input[j]);
			if(!equal(output,[0,0,0,0,0,0,0,0])){
				LSTM.propagate(rate,[0,0,0,0,0,0,0,0]);
			}
		}
		for(var j in input){
			output = LSTM.activate([0,0,0,0,0,0,0,0]);
			if(equal(output,input[j])){
				correct++;
			}else{
				LSTM.propagate(rate,input[j]);
			}
		}
		if((i+1) % 1000 == 0){
			console.log(i+1, correct, Date.now()-start);
			correct = 0;
		}	
	}
}

function test(){  
	test_length = document.getElementById("test-length").value;
	if(test_length == '')
		test_length = 2; 
	input = [];
	output = [];
	prediction = [];
	for(var i = 0; i < test_length; i++){
		input.push([Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random()),Math.round(Math.random())]);
		output.push([0,0,0,0,0,0,0,0]);
	}
	for(var i = 0; i < test_length; i++){
		input.push([0,0,0,0,0,0,0,0]);
		output.push(input[i]);
	}
	for(var i in input){
		prediction.push(LSTM.activate(input[i]));
	}
	drawSequence('input',input);
	drawSequence('output',output);
	drawSequence('prediction',prediction);
}	

function equal(prediction, output) {
  	for (var i in prediction)
   		if (prediction[i] != output[i])
      		return false;
  	return true;
}

function fix_equal(prediction, output) {
  	for (var i in prediction)
   		if (Math.round(prediction[i]) != output[i])
      		return false;
  	return true;
}

function fix_output(arr){
	for(var i in arr)
		arr[i] = Math.round(arr[i]);
	return arr;
}