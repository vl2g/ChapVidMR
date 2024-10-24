## Dataset description
Dataset consists of 4 columns
- id : Unique ID of each video. This ID can be used to get the original video on Youtube using the format: **https://www.youtube.com/watch?v={ID}** 
- query : This is the query that is being asked and needs to be answered using the video
- cleaned_chapters : List consisting of two chapters names which are required to answer the query
- duration_info : The columns containts the duration inforamtion for all the chapters i.e their name, start time, end time and duration

Ex:
```

{
id: IsWgnU71OiY,
query: "How do you connect a momentary push button to a breadboard for use in an electronic circuit?"	,
cleaned_chapters: ['What is a button', 'Circuit Build'],
duration_info: {"Intro": {"start_time": 0, "end_time": 19, "duration": 19}, "What is a button": {"start_time": 19, "end_time": 59, "duration": 40}, "Circuit Build": {"start_time": 59, "end_time": 111, "duration": 52}, "Pullup Resistor": {"start_time": 111, "end_time": 129, "duration": 18}, "Floating GPIO": {"start_time": 129, "end_time": 148, "duration": 19}, "Code Walk Through": {"start_time": 148, "end_time": 222, "duration": 74}, "Internal Pullup Resistor": {"start_time": 222, "end_time": 258, "duration": 36}, "Recap": {"start_time": 258, "end_time": 280, "duration": 22}}
}
```
The link to the Youtube video will be: https://www.youtube.com/watch?v=IsWgnU71OiY. 

