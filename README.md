# AI-BASED-IMAGE-THREAT-DETECTION-
AI-BASED IMAGE THREAT DETECTION  


AI-BASED IDENTIFICATION OF SECURITY THREATS IN DIGITALLY ALTERED IMAGES

This project presents an AI-powered system that detects and evaluates security threats from digitally altered images. It is designed to combat the growing danger of image forgery in critical sectors such as journalism, national security, and law enforcement. The system uses a real-time, self-learning approach powered by Flask and Streamlit, analyzing image anomalies to flag potential threats while improving accuracy over time through user interactions.

🔍 Project Overview

With the rise of AI-generated forgeries and deepfakes, detecting fake images is more important than ever. This project addresses:
- The detection of image manipulations (splicing, cloning, compression artifacts)
- Classification of the security threat level of manipulated images
- Real-time, user-interactive analysis and continuous learning

Objectives

- Train an AI model to detect digitally altered images
- Analyze image characteristics like metadata, color, edges, and compression
- Score threat levels based on manipulation patterns
- Store results dynamically and improve model performance over time
- Build a web-based interface for users to test and visualize results

🛠️ Technologies Used

- Python
- Flask – for backend API handling
- Streamlit – for frontend dashboard and image upload
- OpenCV – for image processing
- NumPy,Pillow – for numerical and image operations
- SQLite – for storing analysis logs
- Matplotlib – for heatmap visualizations


📁 ai-image-threat-detector
│
├── app.py                  Flask backend logic
├── streamlit_app.py        Streamlit frontend interface
├── detection.py            Image manipulation detection functions
├── utils.py                Helper functions (image preprocessing, heatmaps)
├── learn_dataset.csv       Dynamically generated dataset of results
├── requirements.txt        List of required Python packages
└── README.md               Project documentation

How It Works
1. Upload an image through the Streamlit UI.
2. Preprocess and analyze the image using Flask.
3. Detect anomalies such as missing metadata, suspicious edges, JPEG artifacts, and potential splicing.
4. Score the threat level and identify specific manipulation types.
5. Display results with visual cues (e.g., splicing heatmaps).
6. Store the result for ongoing model improvement.

Learning Behavior
Unlike traditional models trained on fixed datasets, this system:
- Starts without a dataset
- Learns dynamically from user-uploaded images
- Adapts to modern manipulation patterns via feedback
- Logs each result into a growing `.csv` file for future AI training

📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

📦 Installation
1. Clone this repo:
   bash
   git clone https://github.com/your-username/ai-image-threat-detector.git
   cd ai-image-threat-detector

2. Create a virtual environment and activate it:
   bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate

3. Install the dependencies:
   bash
   pip install -r requirements.txt

4. Run the Flask backend:
   bash
   python app.py

5. Run the Streamlit frontend:
   bash
   streamlit run streamlit_app.py
  

Sample Features Detected
- EXIF metadata anomalies
- JPEG compression inconsistencies
- Edge density and color outliers
- Image splicing (with heatmap visualization)

✅ Use Cases
- Newsroom image verification
- Social media content moderation
- Law enforcement digital forensics
- AI adversarial threat detection research

🔐 Privacy & Security
- No personal data is stored
- All uploads are anonymized
- Access to stored data is restricted to admins

Future Improvements
- GPU acceleration for faster processing
- Cloud-based deployment and storage
- Facial landmark forgery detection
- Authentication & user login system
- Mobile-friendly interface

👨‍💻 Author
MBRE UTIBE SOLOMON  
CYBERSECURITY FINAL YEAR PROJECT – CALEB UNIVERSITY  
PROJECT SUPERVISED BY: PROF. MOSES K. AREGBESOLA  


License
This project is for academic purposes. For enterprise deployment or contributions, please contact the author.
