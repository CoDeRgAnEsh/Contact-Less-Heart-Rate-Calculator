# Contact-Less-Heart-Rate-Calculator

Monitoring of Heart Rate is almost a necessity in all the telemedicine fields. The cost of external sensors and the presuure appied by the sensors on the fragile skin of children and old age people invokes the demand for contacless monitoring of Heart Rate.

The software developed records the live video of a single person and gives a real time graph with the heart rate value in beats per minute. The values get automatically stored in a csv file for future use. The model also makes a person aware for high heart rate by giving a notification.

Run the below commands to install all the dependencies for the model:
sudo apt-get install python3 python3-pip python3-tk python3-notify2
sudo pip3 install -r requirements.txt


To run the Applications type:
python3 -W ignore HR_v0.1.py

There is an initial wait of about 10s before the program shows an output.
