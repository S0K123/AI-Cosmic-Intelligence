def get_three_js_html(planets_data):
    """
    Generates an enhanced HTML/JS for the Three.js 3D Space Simulation.
    Features: Planet Labels (Sprites), Glowing Orbit Lines, Enhanced Star, Keplerian Motion.
    planets_data: List of dicts with keys: name, radius, orbit_radius, is_habitable
    """
    import json
    planets_json = json.dumps(planets_data)
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AI Cosmos 3D Simulation</title>
        <style>
            body {{ margin: 0; background-color: #000; overflow: hidden; font-family: sans-serif; }}
            canvas {{ width: 100%; height: 100%; display: block; }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        
        <script>
            // Data passed from Python
            const planetsData = {planets_json};

            // --- Scene Setup ---
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);

            // --- Orbit Controls ---
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            camera.position.set(0, 80, 150);
            controls.update();

            // --- Lighting ---
            const ambientLight = new THREE.AmbientLight(0x404040, 1.0); // Brighter ambient
            scene.add(ambientLight);

            // Central Point Light (at the Star)
            const sunLight = new THREE.PointLight(0xffffff, 2.5, 2000);
            sunLight.position.set(0, 0, 0);
            scene.add(sunLight);

            // --- The Host Star (Sun) ---
            const sunGeometry = new THREE.SphereGeometry(10, 64, 64);
            const sunMaterial = new THREE.MeshBasicMaterial({{ color: 0xffcc33 }});
            const sun = new THREE.Mesh(sunGeometry, sunMaterial);
            scene.add(sun);

            // Multi-layered Sun Glow
            for (let i = 1; i <= 3; i++) {{
                const glowGeom = new THREE.SphereGeometry(10 + i * 2, 64, 64);
                const glowMat = new THREE.MeshBasicMaterial({{ 
                    color: 0xffaa00, 
                    transparent: true, 
                    opacity: 0.4 / i 
                }});
                const glowMesh = new THREE.Mesh(glowGeom, glowMat);
                scene.add(glowMesh);
            }}

            // --- Helper: Create Text Label Texture ---
            function createTextTexture(text) {{
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 128;
                
                ctx.fillStyle = 'rgba(0, 0, 0, 0)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                ctx.font = 'Bold 60px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                // Text Shadow for readability
                ctx.shadowColor = 'black';
                ctx.shadowBlur = 10;
                ctx.fillStyle = 'white';
                ctx.fillText(text, 256, 64);
                
                const texture = new THREE.CanvasTexture(canvas);
                return texture;
            }}

            // --- Planets, Orbits, and Labels ---
            const planetObjects = [];

            planetsData.forEach((data, index) => {{
                // 1. Orbit Path (Glowing Line)
                const orbitRadius = data.orbit_radius * 20; //AU scaling
                const orbitCurve = new THREE.EllipseCurve(0, 0, orbitRadius, orbitRadius);
                const orbitPoints = orbitCurve.getPoints(128);
                const orbitGeom = new THREE.BufferGeometry().setFromPoints(orbitPoints);
                const orbitMat = new THREE.LineBasicMaterial({{ 
                    color: data.is_habitable ? 0x00ffff : 0x555555, 
                    transparent: true, 
                    opacity: 0.6 
                }});
                const orbitLine = new THREE.LineLoop(orbitGeom, orbitMat);
                orbitLine.rotation.x = Math.PI / 2;
                scene.add(orbitLine);

                // 2. Planet Mesh
                const pSize = Math.max(1.5, data.radius * 1.2); // Visibility scaling
                const pGeom = new THREE.SphereGeometry(pSize, 32, 32);
                
                // Habitable (Cyan/Green) vs Non-Habitable (Blue/Gray)
                const pColor = data.is_habitable ? 0x00ffcc : 0x4466aa;
                const pMat = new THREE.MeshStandardMaterial({{ 
                    color: pColor,
                    emissive: pColor,
                    emissiveIntensity: 0.3, // Add glow contrast
                    roughness: 0.6,
                    metalness: 0.3
                }});
                const pMesh = new THREE.Mesh(pGeom, pMat);
                scene.add(pMesh);

                // 3. Planet Label (Sprite)
                const labelTexture = createTextTexture(data.name);
                const spriteMat = new THREE.SpriteMaterial({{ map: labelTexture, transparent: true }});
                const labelSprite = new THREE.Sprite(spriteMat);
                labelSprite.scale.set(30, 8, 1); // Fixed readable scale
                scene.add(labelSprite);

                // Keplerian Physics (Speed proportional to 1/sqrt(r))
                const orbitalSpeed = 0.4 / Math.sqrt(data.orbit_radius);
                const initialAngle = Math.random() * Math.PI * 2;

                planetObjects.push({{
                    mesh: pMesh,
                    label: labelSprite,
                    radius: orbitRadius,
                    speed: orbitalSpeed,
                    angle: initialAngle,
                    labelOffset: pSize + 5 // Float above planet
                }});
            }});

            // --- Background: 10,000 Stars ---
            const starGeom = new THREE.BufferGeometry();
            const starMat = new THREE.PointsMaterial({{ color: 0xffffff, size: 0.2, transparent: true, opacity: 0.8 }});
            const starVerts = [];
            for (let i = 0; i < 10000; i++) {{
                const x = (Math.random() - 0.5) * 3000;
                const y = (Math.random() - 0.5) * 3000;
                const z = (Math.random() - 0.5) * 3000;
                starVerts.push(x, y, z);
            }}
            starGeom.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
            const stars = new THREE.Points(starGeom, starMat);
            scene.add(stars);

            // --- Animation Loop ---
            function animate() {{
                requestAnimationFrame(animate);

                // Rotate Sun
                sun.rotation.y += 0.002;

                // Move Planets and Update Labels
                planetObjects.forEach(obj => {{
                    obj.angle += obj.speed * 0.01;
                    const x = Math.cos(obj.angle) * obj.radius;
                    const z = Math.sin(obj.angle) * obj.radius;
                    
                    // Sync Planet Position
                    obj.mesh.position.set(x, 0, z);
                    obj.mesh.rotation.y += 0.015;
                    
                    // Sync Label Position (floating above)
                    obj.label.position.set(x, obj.labelOffset, z);
                }});

                controls.update();
                renderer.render(scene, camera);
            }}

            // Handle Window Resize
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});

            animate();
        </script>
    </body>
    </html>
    """
    return html_template
