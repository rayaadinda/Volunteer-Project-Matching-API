"""
Volunteer-Project Matching API
Indonesian Heritage Society

Flask API that provides volunteer-project matching services with cosine similarity.
Handles flexible numbers of volunteers and projects.
"""

from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Any
import warnings
import logging
from datetime import datetime
import traceback

# Try to import CORS, but continue without it if not available
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not available. CORS will be disabled.")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)  # Enable Cross-Origin Resource Sharing

class VolunteerProjectMatcher:
    def __init__(self):
        """Initialize the matcher with duration mappings and feature collections."""
        self.duration_mapping = {
            "1 week": 1,
            "2 weeks": 2, 
            "1 month": 3,
            "1-3 months": 3.5,
            "3-6 months": 4.5,
            "> 1 month": 4,
            "6 months": 6,
            "1 year": 12
        }
        
        self.opportunity_features = set()
        self.personality_features = set()
        
    def extract_features_from_data(self, volunteers: List[Dict], projects: List[Dict]):
        """Extract all possible features from the datasets for consistent vectorization."""
        # Reset features for each request
        self.opportunity_features = set()
        self.personality_features = set()
        
        # Extract opportunity features from volunteers
        for volunteer in volunteers:
            opportunities = volunteer.get('volunteer_opportunities', {})
            self._flatten_opportunities(opportunities, '')
            
        # Extract opportunity features from projects
        for project in projects:
            for tag in project.get('opportunity_tags', []):
                # Convert project tags to match volunteer structure
                if ' - ' in tag:
                    category, skill = tag.split(' - ', 1)
                    category = category.lower().replace(' ', '_')
                    skill = skill.lower().replace(' ', '_').replace('-', '_')
                    feature_name = f"{category}_{skill}"
                    self.opportunity_features.add(feature_name)
                else:
                    self.opportunity_features.add(tag.lower().replace(' ', '_').replace('-', '_'))
        
        # Extract personality features
        for volunteer in volunteers:
            enneagram = volunteer.get('enneagram', {})
            for personality in enneagram.keys():
                self.personality_features.add(personality)
                
        for project in projects:
            for personality in project.get('personality_fit', []):
                self.personality_features.add(personality.lower())
        
        logger.info(f"Extracted {len(self.opportunity_features)} opportunity features")
        logger.info(f"Extracted {len(self.personality_features)} personality features")
    
    def _flatten_opportunities(self, opportunities: Dict, prefix: str = ''):
        """Recursively flatten volunteer opportunities structure."""
        for key, value in opportunities.items():
            if isinstance(value, dict):
                new_prefix = f"{prefix}_{key}" if prefix else key
                self._flatten_opportunities(value, new_prefix)
            elif isinstance(value, bool):
                feature_name = f"{prefix}_{key}" if prefix else key
                self.opportunity_features.add(feature_name)
    
    def vectorize_volunteer(self, volunteer: Dict) -> np.ndarray:
        """Convert volunteer profile to feature vector."""
        vector = []
        
        # 1. Volunteer opportunities (one-hot encoded)
        opportunities = volunteer.get('volunteer_opportunities', {})
        flattened_opps = self._flatten_volunteer_opportunities(opportunities)
        
        for feature in sorted(self.opportunity_features):
            vector.append(1 if flattened_opps.get(feature, False) else 0)
        
        # 2. Enneagram (keep only selected ones)
        enneagram = volunteer.get('enneagram', {})
        selected_personalities = [k for k, v in enneagram.items() if v]
        
        for personality in sorted(self.personality_features):
            if personality in selected_personalities:
                vector.append(1)
            else:
                vector.append(0)
        
        # 3. Duration preference (ordinal)
        duration = volunteer.get('preferred_duration', '1 month')
        duration_score = self.duration_mapping.get(duration, 3)
        vector.append(duration_score)
        
        return np.array(vector, dtype=float)
    
    def _flatten_volunteer_opportunities(self, opportunities: Dict, prefix: str = '') -> Dict:
        """Flatten volunteer opportunities to match feature extraction."""
        flattened = {}
        for key, value in opportunities.items():
            if isinstance(value, dict):
                new_prefix = f"{prefix}_{key}" if prefix else key
                flattened.update(self._flatten_volunteer_opportunities(value, new_prefix))
            elif isinstance(value, bool):
                feature_name = f"{prefix}_{key}" if prefix else key
                flattened[feature_name] = value
        return flattened
    
    def vectorize_project(self, project: Dict) -> np.ndarray:
        """Convert project profile to feature vector."""
        vector = []
        
        # 1. Opportunity tags (one-hot encoded)
        project_opportunities = set()
        for tag in project.get('opportunity_tags', []):
            if ' - ' in tag:
                category, skill = tag.split(' - ', 1)
                category = category.lower().replace(' ', '_')
                skill = skill.lower().replace(' ', '_').replace('-', '_')
                feature_name = f"{category}_{skill}"
                project_opportunities.add(feature_name)
            else:
                project_opportunities.add(tag.lower().replace(' ', '_').replace('-', '_'))
        
        for feature in sorted(self.opportunity_features):
            vector.append(1 if feature in project_opportunities else 0)
        
        # 2. Personality fit (one-hot encoded)
        project_personalities = set(p.lower() for p in project.get('personality_fit', []))
        
        for personality in sorted(self.personality_features):
            vector.append(1 if personality in project_personalities else 0)
        
        # 3. Duration (ordinal)
        duration = project.get('estimated_duration', '1 month')
        duration_score = self.duration_mapping.get(duration, 3)
        vector.append(duration_score)
        
        return np.array(vector, dtype=float)
    
    def check_language_compatibility(self, volunteer: Dict, project: Dict) -> bool:
        """Check if volunteer speaks all required languages for the project."""
        volunteer_languages = set(lang.lower().strip() for lang in volunteer.get('languages', []))
        required_languages = project.get('required_languages', [])
        
        if not required_languages:
            return True
            
        required_languages = set(lang.lower().strip() for lang in required_languages)
        return required_languages.issubset(volunteer_languages)
    
    def match_volunteers_to_projects(self, volunteers: List[Dict], projects: List[Dict], 
                                   top_k: int = 4) -> List[Dict]:
        """
        Match each volunteer to top K most suitable projects.
        
        Args:
            volunteers: List of volunteer profiles
            projects: List of project profiles  
            top_k: Number of top matches to return per volunteer
            
        Returns:
            List of matching results
        """
        if not volunteers or not projects:
            logger.warning("Empty volunteers or projects list provided")
            return []
        
        results = []
        
        # Extract features from all data
        self.extract_features_from_data(volunteers, projects)
        
        # Vectorize all volunteers and projects
        volunteer_vectors = []
        project_vectors = []
        
        logger.info(f"Vectorizing {len(volunteers)} volunteers...")
        for volunteer in volunteers:
            try:
                vector = self.vectorize_volunteer(volunteer)
                volunteer_vectors.append(vector)
            except Exception as e:
                logger.error(f"Error vectorizing volunteer {volunteer.get('id', 'unknown')}: {e}")
                # Add zero vector as fallback
                volunteer_vectors.append(np.zeros(len(sorted(self.opportunity_features)) + 
                                                len(sorted(self.personality_features)) + 1))
        
        logger.info(f"Vectorizing {len(projects)} projects...")
        for project in projects:
            try:
                vector = self.vectorize_project(project)
                project_vectors.append(vector)
            except Exception as e:
                logger.error(f"Error vectorizing project {project.get('project_id', 'unknown')}: {e}")
                # Add zero vector as fallback
                project_vectors.append(np.zeros(len(sorted(self.opportunity_features)) + 
                                              len(sorted(self.personality_features)) + 1))
        
        if not volunteer_vectors or not project_vectors:
            logger.error("Failed to vectorize volunteers or projects")
            return []
        
        volunteer_matrix = np.array(volunteer_vectors)
        project_matrix = np.array(project_vectors)
        
        # Normalize vectors for better cosine similarity
        volunteer_matrix = normalize(volunteer_matrix, norm='l2')
        project_matrix = normalize(project_matrix, norm='l2')
        
        logger.info(f"Computing similarities for {len(volunteers)} volunteers...")
        
        # For each volunteer, find top matches
        for i, volunteer in enumerate(volunteers):
            try:
                volunteer_vector = volunteer_matrix[i:i+1]
                
                # Filter projects by language compatibility
                eligible_projects = []
                eligible_vectors = []
                
                for j, project in enumerate(projects):
                    if self.check_language_compatibility(volunteer, project):
                        eligible_projects.append((j, project))
                        eligible_vectors.append(project_matrix[j])
                
                if not eligible_projects:
                    # If no eligible projects, use all projects
                    logger.warning(f"No language-compatible projects for volunteer {volunteer.get('id', i)}")
                    eligible_projects = [(j, project) for j, project in enumerate(projects)]
                    eligible_vectors = project_matrix
                
                if eligible_vectors:
                    eligible_matrix = np.array(eligible_vectors)
                    
                    # Compute cosine similarity
                    similarities = cosine_similarity(volunteer_vector, eligible_matrix)[0]
                    
                    # Get top K matches
                    top_indices = np.argsort(similarities)[::-1][:top_k]
                    
                    top_matches = []
                    for idx in top_indices:
                        if idx < len(eligible_projects):
                            project_idx, project = eligible_projects[idx]
                            score = similarities[idx]
                            
                            top_matches.append({
                                "project_id": project.get('project_id', f'PROJ-{project_idx}'),
                                "project_name": project.get('project_name', 'Unknown Project'),
                                "match_score": round(float(score), 3)
                            })
                    
                    results.append({
                        "volunteer_id": volunteer.get('id', f'VOL-{i}'),
                        "name": volunteer.get('name', 'Unknown Volunteer'),
                        "top_matches": top_matches
                    })
                
                # Progress logging for large datasets
                if (i + 1) % 100 == 0 or i == len(volunteers) - 1:
                    logger.info(f"Processed {i + 1}/{len(volunteers)} volunteers")
                    
            except Exception as e:
                logger.error(f"Error matching volunteer {volunteer.get('id', i)}: {e}")
                # Add empty result to maintain order
                results.append({
                    "volunteer_id": volunteer.get('id', f'VOL-{i}'),
                    "name": volunteer.get('name', 'Unknown Volunteer'),
                    "top_matches": [],
                    "error": str(e)
                })
        
        return results

# Initialize the matcher
matcher = VolunteerProjectMatcher()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Volunteer-Project Matching API",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/match', methods=['POST'])
def match_volunteers():
    """
    Main matching endpoint.
    
    Expected JSON payload:
    {
        "volunteers": [...],  // Array of volunteer objects
        "projects": [...],    // Array of project objects
        "top_k": 4           // Optional: number of matches per volunteer (default: 4)
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON",
                "status": "error"
            }), 400
        
        data = request.get_json()
        
        # Extract data
        volunteers = data.get('volunteers', [])
        projects = data.get('projects', [])
        top_k = data.get('top_k', 4)
        
        # Validate inputs
        if not isinstance(volunteers, list) or not isinstance(projects, list):
            return jsonify({
                "error": "volunteers and projects must be arrays",
                "status": "error"
            }), 400
        
        if len(volunteers) == 0:
            return jsonify({
                "error": "No volunteers provided",
                "status": "error"
            }), 400
            
        if len(projects) == 0:
            return jsonify({
                "error": "No projects provided", 
                "status": "error"
            }), 400
        
        if not isinstance(top_k, int) or top_k < 1:
            return jsonify({
                "error": "top_k must be a positive integer",
                "status": "error"
            }), 400
        
        # Log request info
        logger.info(f"Matching request: {len(volunteers)} volunteers, {len(projects)} projects, top_k={top_k}")
        
        # Perform matching
        start_time = datetime.now()
        results = matcher.match_volunteers_to_projects(volunteers, projects, top_k)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        total_matches = sum(len(result.get('top_matches', [])) for result in results)
        avg_score = 0
        if total_matches > 0:
            all_scores = []
            for result in results:
                for match in result.get('top_matches', []):
                    all_scores.append(match.get('match_score', 0))
            avg_score = np.mean(all_scores) if all_scores else 0
        
        return jsonify({
            "status": "success",
            "data": results,
            "metadata": {
                "volunteers_processed": len(volunteers),
                "projects_available": len(projects),
                "total_matches_generated": total_matches,
                "average_match_score": round(float(avg_score), 3),
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": datetime.now().isoformat(),
                "top_k": top_k
            }
        })
        
    except Exception as e:
        logger.error(f"Error in match endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/match/batch', methods=['POST'])
def batch_match():
    """
    Batch matching endpoint for very large datasets.
    Processes volunteers in batches to handle memory constraints.
    
    Expected JSON payload:
    {
        "volunteers": [...],     // Array of volunteer objects
        "projects": [...],       // Array of project objects  
        "top_k": 4,             // Optional: number of matches per volunteer
        "batch_size": 100       // Optional: batch size for processing (default: 100)
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON",
                "status": "error"
            }), 400
        
        data = request.get_json()
        volunteers = data.get('volunteers', [])
        projects = data.get('projects', [])
        top_k = data.get('top_k', 4)
        batch_size = data.get('batch_size', 100)
        
        if not volunteers or not projects:
            return jsonify({
                "error": "volunteers and projects are required",
                "status": "error"
            }), 400
        
        logger.info(f"Batch matching: {len(volunteers)} volunteers, {len(projects)} projects, batch_size={batch_size}")
        
        # Process in batches
        all_results = []
        start_time = datetime.now()
        
        for i in range(0, len(volunteers), batch_size):
            batch_volunteers = volunteers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: volunteers {i+1}-{min(i+batch_size, len(volunteers))}")
            
            batch_results = matcher.match_volunteers_to_projects(batch_volunteers, projects, top_k)
            all_results.extend(batch_results)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        total_matches = sum(len(result.get('top_matches', [])) for result in all_results)
        
        return jsonify({
            "status": "success",
            "data": all_results,
            "metadata": {
                "volunteers_processed": len(volunteers),
                "projects_available": len(projects),
                "total_matches_generated": total_matches,
                "batch_size": batch_size,
                "batches_processed": (len(volunteers) + batch_size - 1) // batch_size,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": datetime.now().isoformat(),
                "top_k": top_k
            }
        })
        
    except Exception as e:
        logger.error(f"Error in batch match endpoint: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/validate', methods=['POST'])
def validate_data():
    """
    Validate volunteer and project data structure.
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON",
                "status": "error"
            }), 400
        
        data = request.get_json()
        volunteers = data.get('volunteers', [])
        projects = data.get('projects', [])
        
        validation_results = {
            "volunteers": {
                "count": len(volunteers),
                "valid": 0,
                "errors": []
            },
            "projects": {
                "count": len(projects),
                "valid": 0,
                "errors": []
            }
        }
        
        # Validate volunteers
        required_volunteer_fields = ['id', 'name', 'languages']
        for i, volunteer in enumerate(volunteers):
            is_valid = True
            for field in required_volunteer_fields:
                if field not in volunteer:
                    validation_results["volunteers"]["errors"].append(f"Volunteer {i}: Missing required field '{field}'")
                    is_valid = False
            if is_valid:
                validation_results["volunteers"]["valid"] += 1
        
        # Validate projects
        required_project_fields = ['project_id', 'project_name']
        for i, project in enumerate(projects):
            is_valid = True
            for field in required_project_fields:
                if field not in project:
                    validation_results["projects"]["errors"].append(f"Project {i}: Missing required field '{field}'")
                    is_valid = False
            if is_valid:
                validation_results["projects"]["valid"] += 1
        
        return jsonify({
            "status": "success",
            "validation": validation_results
        })
        
    except Exception as e:
        logger.error(f"Error in validate endpoint: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

# Handler for Vercel
app.debug = False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
