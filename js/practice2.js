var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

var LSTM = new Architect.LSTM(96,30,96);
var iterations = 100000;
var rate = .1;
var success = 0.95;
var input = [],
	output = [];
var trial, correct, criterion;
var start;

function read_and_train(){  
	var text = document.getElementById("textarea").value;
	for(var i = 0; i < (text.length - 1); i++){
		input = char_to_binary(text.charAt(i));
		output = char_to_binary(text.charAt(i + 1));
		trial = 1;
		correct = 0;
		criterion = 0;
		start = Date.now();
		while(trial < iterations && criterion < success){
			var prediction = LSTM.activate(input);
			if(equal(prediction, output)){
				correct++;
			}else{
				LSTM.propagate(rate, output);
			}
			if(trial % 50 == 0){
				criterion = correct / 50;
				correct = 0;
			}
			trial++;
		}
		console.log(text.charAt(i), trial, criterion * 100, Date.now() - start);
		test();
	}
}	

function test(){
	var text = document.getElementById("text").value;
	input = char_to_binary(text.charAt(0));
	output = fix_output(LSTM.activate(input));
	console.log("output: "+binary_to_char(output));
}

function char_to_binary(c){
	var arr = new Array(96);
	var c_ascii = c.charCodeAt(0);
	for(var i = 0; i < arr.length; i++)
		arr[i] = 0;
	if(c_ascii == 10){
		arr[0] = 1;
	}else{
		arr[c_ascii - 31] = 1;
	}
	return arr;
}

function binary_to_char(arr){
	var index = arr.indexOf(1);
	if(index == 0)
		return String.fromCharCode(10);
	return String.fromCharCode(index + 31);
}

function equal(prediction, output) {
  	for (var i in prediction)
   		if (Math.round(prediction[i]) != output[i])
      		return false;
  	return true;
};

function fix_output(arr){
	for(var i in arr)
		arr[i] = Math.round(arr[i]);
	return arr;
}