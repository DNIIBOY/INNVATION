#include "tracker.h"
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

PeopleTracker::PeopleTracker(int maxMissingFrames, float maxDistance, float topThreshold, float bottomThreshold)
    : nextId(0), maxMissingFrames(maxMissingFrames), maxDistance(maxDistance),
      topThreshold(topThreshold), bottomThreshold(bottomThreshold), movementCallback(nullptr) {
#ifdef DEBUG
    cout << "Tracker initialized with maxMissingFrames=" << maxMissingFrames 
         << ", maxDistance=" << maxDistance << endl;
#endif
}

void PeopleTracker::update(const vector<Detection>& detections, int frameHeight) {
    vector<TrackedPerson> newPeople;
    
    // Process each detection
    for (const auto& det : detections) {
        // Only track people
        if (det.classId != "person") continue;
        
        // Convert detection to position and size
        Position pos = {
            det.box.x + det.box.width / 2,
            det.box.y + det.box.height / 2
        };
        BoxSize size = {
            det.box.width,
            det.box.height
        };
        
        // Find closest matching person
        TrackedPerson closestPerson;
        float minDistance = maxDistance + 1;  // Initialize to more than maximum
        int closestPersonIndex = -1;
        
        for (size_t i = 0; i < people.size(); i++) {
            const auto& person = people[i];
            float distance = sqrt(pow(person.pos.x - pos.x, 2) + pow(person.pos.y - pos.y, 2));
            
            if (distance < minDistance) {
                minDistance = distance;
                closestPersonIndex = i;
            }
        }
        
        if (closestPersonIndex >= 0 && minDistance < maxDistance) {
            // Update existing person
            TrackedPerson updatedPerson = people[closestPersonIndex];
            updatedPerson.update(pos, size, det.confidence);
            
            // Check for movements between zones
            detectMovements(updatedPerson, frameHeight);
            
            newPeople.push_back(updatedPerson);
            people.erase(people.begin() + closestPersonIndex);
        } else {
            // Create new tracked person
            TrackedPerson newPerson(nextId++, pos, size, det.confidence, frameHeight);
            newPeople.push_back(newPerson);
        }
    }
    
    // Handle missing people (not matched with any detection)
    for (auto& missingPerson : people) {
        missingPerson.missingFrames++;
        if (missingPerson.missingFrames < maxMissingFrames) {
            newPeople.push_back(missingPerson);
        } else {
            // Person has disappeared - check if they crossed a boundary
            detectMovements(missingPerson, frameHeight);
#ifdef DEBUG
            cout << "Person ID " << missingPerson.id << " has disappeared" << endl;
#endif
        }
    }
    
    // Update the people list
    people = newPeople;
    
#ifdef DEBUG
    cout << "Tracking updated: " << people.size() << " people tracked" << endl;
#endif
}

void PeopleTracker::detectMovements(const TrackedPerson& person, int frameHeight) {
    if (person.history.size() < 5) return;  // Need enough history to determine movement
    
    // Get first and last positions in history
    const Position& start = person.history.front();
    const Position& end = person.history.back();
    
    int startY = start.y;
    int endY = end.y;
    
    // Check if person moved from top to bottom
    if (person.fromTop && endY > frameHeight * bottomThreshold) {
        if (movementCallback) {
            movementCallback(person, "exit");
        }
#ifdef DEBUG
        cout << "Person ID " << person.id << " moved from top to bottom" << endl;
#endif
    }
    
    // Check if person moved from bottom to top
    if (person.fromBottom && endY < frameHeight * topThreshold) {
        if (movementCallback) {
            movementCallback(person, "enter");
        }
#ifdef DEBUG
        cout << "Person ID " << person.id << " moved from bottom to top" << endl;
#endif
    }
}

void PeopleTracker::draw(Mat& frame) {
    for (const auto& person : people) {
        // Draw bounding box
        rectangle(frame, person.getBoundingBox(), person.color, 2);
        
        // Draw ID and entry point
        string label = "ID: " + to_string(person.id);
        if (person.fromTop) {
            label += " (Top)";
        } else if (person.fromBottom) {
            label += " (Bottom)";
        }
        
        int y = max(person.pos.y - person.size.height / 2 - 10, 15);
        putText(frame, label, Point(person.pos.x - person.size.width / 2, y), 
                FONT_HERSHEY_SIMPLEX, 0.5, person.color, 2);
                
        // Draw movement trail/history
        if (person.history.size() > 1) {
            for (size_t i = 1; i < person.history.size(); i++) {
                // Make trail gradually fade
                float alpha = static_cast<float>(i) / person.history.size();
                Scalar trailColor = person.color * alpha;
                
                circle(frame, Point(person.history[i-1].x, person.history[i-1].y), 2, trailColor, -1);
                line(frame, Point(person.history[i-1].x, person.history[i-1].y),
                     Point(person.history[i].x, person.history[i].y), trailColor, 1);
            }
            
            // Draw current position
            circle(frame, Point(person.history.back().x, person.history.back().y), 4, person.color, -1);
        }
    }
    
    // Draw entry/exit zones
    int topZoneY = frame.rows * topThreshold;
    int bottomZoneY = frame.rows * bottomThreshold;
    
    line(frame, Point(0, topZoneY), Point(frame.cols, topZoneY), Scalar(0, 255, 255), 1);
    line(frame, Point(0, bottomZoneY), Point(frame.cols, bottomZoneY), Scalar(0, 255, 255), 1);
    
    // Add labels for zones
    putText(frame, "Entry zone", Point(10, topZoneY - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
    putText(frame, "Exit zone", Point(10, bottomZoneY + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
}

const vector<TrackedPerson>& PeopleTracker::getTrackedPeople() const {
    return people;
}

void PeopleTracker::setMovementCallback(MovementCallback callback) {
    movementCallback = callback;
}