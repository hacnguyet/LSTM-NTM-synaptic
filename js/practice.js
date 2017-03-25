var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

var LSTM = new Architect.LSTM(3,10,3);
var iterations = 100000;
var rate = .1;
var lastInput = 0;
var input, input_test, output_test, output, mem;

function train(){
	iterations = document.getElementById("iterations").value;
	if(iterations == '')
		iterations = 100000; 
	for(var i = 0; i < iterations; i++){
		lastInput = 0;
		mem = [0,0,0];
		for(var z = 0; z < 10; z++){
			//Create sequence
			j = 0;
			input = [0,0,0];
			if(z != 9){
				var j = Math.floor((Math.random() * 3) + 1);
				if(j == 1)
					input = [1,0,0];
				if(j == 2)
					input = [0,1,0];
				if(j == 3)
					input = [0,0,1];
			}

			//Compute output
			if(lastInput == j){
				if(j == 1)
					mem = [1,0,0];
				if(j == 2)
					mem = [0,1,0];
				if(j == 3)
					mem = [0,0,1];
			}
			output = mem;
			if(j == 1 || j == 2 || j == 3)
				output = [0,0,0];
			lastInput = j;
			//Train
		    var res = LSTM.activate(input);
		    if (Math.round(res)[0] != output[0] && Math.round(res)[1] != output[1] && Math.round(res)[2] != output[2])
		    	LSTM.propagate(rate, output);
		}
		if(i % 10000 == 0){
	    	console.log(i, input, res, output);
	    }	
	 }  
}

function test(){  
	input_test = document.getElementById("input").value;
	var arr = new Array(3);
	arr[0] = parseInt(input_test.charAt(0));
	arr[1] = parseInt(input_test.charAt(1));
	arr[2] = parseInt(input_test.charAt(2));
	output_test = fix_output(LSTM.activate(arr));
	document.getElementById("output").value = output_test[0] + '' + output_test[1]  + '' + output_test[2];
	console.log(arr, output_test);
	// console.log(LSTM.activate([0,1,0]));
	// console.log(LSTM.activate([0,1,0]));
	// console.log(LSTM.activate([0,0,0]));
}	

function fix_output(arr){
	for(var i in arr)
		arr[i] = Math.round(arr[i]);
	return arr;
}