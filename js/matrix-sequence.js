function drawSequence(tableId, sequence){
	for(var j=0; j < sequence.length; j++){
		var rowCount = $('#'+tableId+' tr').length;
		for(var i=0; i < (sequence[j].length - rowCount); i++){
			$('#'+tableId).append("<tr></tr>");
		}
		var rows = $('tr', '#'+tableId);
		for(var i=0; i < sequence[j].length; i++){
			rows.eq(i).append('<td style="'+convertColor(sequence[j][i])+'"></td>');
		}
	}
}

function convertColor(f){
	return 'background-color: hsl('+(240-f*240)+',100%,55%);';
}

function clear_table(){
	$('#input tr').remove();
	$('#output tr').remove();
	$('#prediction tr').remove();
}