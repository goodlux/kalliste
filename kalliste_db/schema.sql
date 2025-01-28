CREATE TABLE tag (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag TEXT NOT NULL UNIQUE
);

CREATE TABLE image (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    photoshoot TEXT NOT NULL,
    photoshoot_date DATE,
    photoshoot_location TEXT,
    person_name TEXT,
    source_type TEXT,
    lr_rating INTEGER,
    creation_date TIMESTAMP,
    region_type TEXT,
    nima_technical_score REAL,
    nima_assessment_technical TEXT,
    nima_aesthetic_score REAL,
    nima_assessment_aesthetic TEXT,
    nima_overall_score REAL,
    nima_assessment_overall TEXT,
    nima_calc_average REAL,
    assessment TEXT
);

-- Indexes for common queries and foreign key relationships
CREATE INDEX idx_image_person_name ON image(person_name);
CREATE INDEX idx_image_photoshoot ON image(photoshoot);
CREATE INDEX idx_image_photoshoot_date ON image(photoshoot_date);
CREATE INDEX idx_image_lr_rating ON image(lr_rating);
CREATE INDEX idx_image_nima_calc_average ON image(nima_calc_average);
CREATE INDEX idx_image_region_type ON image(region_type);

-- Compound indexes for multi-field queries
CREATE INDEX idx_image_person_quality ON image(person_name, nima_calc_average, lr_rating);
CREATE INDEX idx_image_photoshoot_date_person ON image(photoshoot_date, person_name);

CREATE TABLE image_tag (
    image_id INTEGER,
    tag_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES image(image_id),
    FOREIGN KEY (tag_id) REFERENCES tag(tag_id),
    PRIMARY KEY (image_id, tag_id)
);

-- Index for tag lookups (though typically handled by the primary key)
CREATE INDEX idx_image_tag_tag_id ON image_tag(tag_id);

CREATE TABLE input_queue (
   queue_id INTEGER PRIMARY KEY AUTOINCREMENT, 
   file_path TEXT NOT NULL UNIQUE,
   status TEXT NOT NULL,
   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
   last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for queue status queries
CREATE INDEX idx_input_queue_status ON input_queue(status);
CREATE INDEX idx_input_queue_dates ON input_queue(created_at, last_updated);