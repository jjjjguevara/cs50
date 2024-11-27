import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = 'http://localhost:5001/api'

def list_existing_topics():
    """List all existing topics in the system"""
    try:
        response = requests.get(f'{BASE_URL}/topics')
        logger.info(f"Existing topics: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        return []

def create_room_acoustics_topic():
    """Create the room acoustics topic"""
    # First, let's see what topics exist
    logger.info("Checking existing topics...")
    existing_topics = list_existing_topics()

    topic_data = {
        "title": "Room Acoustics: Theory and Practice in Audio Engineering",
        "type": "topic",
        "content": """
        <prolog>
            <metadata>
                <othermeta name="journal" content="Journal of Audio Engineering and Acoustics"/>
                <othermeta name="doi" content="10.1000/jaes.2024.11.001"/>
                <othermeta name="publication-date" content="2024-11-27"/>
                <author>Dr. Jane Smith, Ph.D.</author>
                <author>Prof. Peter Anderson</author>
                <institution>Audio Engineering Institute</institution>
                <category>Room Acoustics</category>
                <category>Studio Design</category>
                <keywords>
                    <keyword>modal response</keyword>
                    <keyword>room modes</keyword>
                    <keyword>standing waves</keyword>
                    <keyword>acoustic treatment</keyword>
                </keywords>
            </metadata>
        </prolog>
        <abstract>
            <shortdesc>A comprehensive analysis of room acoustics principles.</shortdesc>
            <p>This paper examines room acoustics principles and their implementation.</p>
        </abstract>
        <body>
            <section id="introduction">
                <title>Introduction</title>
                <p>Room acoustics is fundamental to sound quality.</p>
            </section>
        </body>
        """
    }

    try:
        logger.info("Creating room acoustics topic...")
        response = requests.post(
            f'{BASE_URL}/topics',
            json=topic_data,
            headers={'Content-Type': 'application/json'}
        )

        logger.info(f"Creation Status Code: {response.status_code}")
        logger.info(f"Creation Response: {json.dumps(response.json(), indent=2)}")

        # Check if creation was successful
        if response.status_code == 200:
            # Get the created topic's path
            created_topic = response.json()
            logger.info(f"Topic created at: {created_topic.get('path')}")

            # List topics again to confirm creation
            logger.info("Checking updated topic list...")
            updated_topics = list_existing_topics()

            return created_topic
    except Exception as e:
        logger.error(f"Error creating topic: {e}")
        return None

if __name__ == '__main__':
    logger.info("Starting room acoustics topic creation...")
    result = create_room_acoustics_topic()
    if result:
        # Get the topic ID from the path
        topic_path = result.get('path', '')
        topic_id = topic_path.split('/')[-1].replace('.dita', '')

        logger.info("Successfully created room acoustics article")
        logger.info("Topic details:")
        logger.info(f"Path: {result.get('path')}")
        logger.info(f"View URL: {BASE_URL}/view/{topic_id}")

        # Try to view the topic
        view_response = requests.get(f'{BASE_URL}/view/{topic_id}')
        logger.info(f"View Status Code: {view_response.status_code}")
        if view_response.status_code != 200:
            logger.error(f"Error viewing topic: {view_response.text}")
    else:
        logger.error("Failed to create room acoustics article")
