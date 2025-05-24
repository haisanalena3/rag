# rag-app Documentation

## Overview
The `rag-app` project is designed to collect data from specified URLs and utilize that data for question and answer sessions through the Gemini API. The application employs a retrieval-augmented generation (RAG) approach to provide contextually relevant answers based on the collected data.

## Project Structure
```
rag-app
├── src
│   ├── collect.py       # Script to collect data from URLs
│   ├── rag.py           # Script for Q&A using the Gemini API
│   ├── config.py        # Configuration settings and environment variables
│   └── db               # Directory for storing collected data
├── .env                 # Environment variables for configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd rag-app
   ```

2. **Create a Virtual Environment**
   It is recommended to create a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   Install the required Python packages listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory with the following structure:
   ```
   URLS=<comma-separated list of URLs>
   GEMINI_API_KEY=<your_gemini_api_key>
   MODEL_NAME=<your_model_name>
   ```

5. **Run the Data Collection Script**
   Execute the `collect.py` script to gather data from the specified URLs.
   ```bash
   python src/collect.py
   ```

6. **Interact with the RAG System**
   Use the `rag.py` script to ask questions and receive answers based on the collected data.
   ```bash
   python src/rag.py
   ```

## Usage
- After running the `collect.py` script, the collected data will be stored in the `src/db` directory.
- When running `rag.py`, you will be prompted to input questions. The script will utilize the Gemini API to provide answers based on the context of the collected data.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.