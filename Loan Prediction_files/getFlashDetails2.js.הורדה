$(document).ready(function(){let currentHost=window.location.hostname;data={currentHost:currentHost}
$.ajax({url:'https://automation.analyticsvidhya.com/flashstrip/getstrip/',data:data,type:"POST",timeout:3000,async:false,success:function(response){if(response){$('#sale-strip').show()
$("#sale-strip p").html(response['text'])
$("#sale-strip a").html(response['button_text'])
$("#href_id").attr('href',response['url'])
if(response['timer_end_date']){$('#clockdiv').show();deadline=moment(response['timer_end_date'],moment.ISO_8601).toDate()
console.log(deadline)}}},})})
if(deadline!=null)
var deadline=new Date("sep 25, 2020 23:59:59").getTime();var x=setInterval(function(){var now=new Date().getTime();var differnce_in_time=deadline-now;var days=Math.floor(differnce_in_time/(1000*60*60*24));var hours=Math.floor((differnce_in_time%(1000*60*60*24))/(1000*60*60));var minutes=Math.floor((differnce_in_time%(1000*60*60))/(1000*60));var seconds=Math.floor((differnce_in_time%(1000*60))/1000);if(seconds<10){seconds='0'+seconds;}
if(minutes<10){minutes='0'+minutes;}
if(hours<10){hours='0'+hours;}
if(days<10){days='0'+days;}
document.getElementById("day").innerHTML=days;document.getElementById("hour").innerHTML=hours;document.getElementById("minute").innerHTML=minutes;document.getElementById("second").innerHTML=seconds;if(differnce_in_time<0){clearInterval(x);document.getElementById("clockdiv").style.display="none";document.getElementById("day").innerHTML='0';document.getElementById("hour").innerHTML='0';document.getElementById("minute").innerHTML='0';document.getElementById("second").innerHTML='0';}},1000);