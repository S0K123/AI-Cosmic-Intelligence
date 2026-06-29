
import json


def get_three_js_html(planets_data, star_info=None):
    """
    Generates HTML/JS for Three.js 3D Space Simulation
    """
    planets_json = json.dumps(planets_data)
    star_json = json.dumps(star_info or {})

    return f"""
&lt;!DOCTYPE html&gt;
&lt;html lang="en"&gt;
&lt;head&gt;
    &lt;meta charset="UTF-8"&gt;
    &lt;meta name="viewport" content="width=device-width, initial-scale=1.0"&gt;
    &lt;title&gt;AI Cosmos 3D Simulation&lt;/title&gt;
    &lt;style&gt;
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background-color: #000;
            overflow: hidden;
            font-family: 'Arial', sans-serif;
        }}
        canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        #info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(10, 10, 30, 0.95);
            color: #fff;
            padding: 15px 20px;
            border-radius: 10px;
            max-width: 320px;
            border: 1px solid #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.35);
            display: none;
            z-index: 100;
        }}
        #info-panel h3 {{
            margin: 0 0 12px 0;
            color: #00d4ff;
        }}
        #info-panel p {{
            margin: 6px 0;
            font-size: 14px;
            line-height: 1.5;
        }}
        #close-info {{
            position: absolute;
            top: 8px;
            right: 12px;
            background: none;
            border: none;
            color: #fff;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
        }}
        #star-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(10, 10, 30, 0.95);
            color: #fff;
            padding: 15px 20px;
            border-radius: 10px;
            max-width: 280px;
            border: 1px solid #ffcc33;
            box-shadow: 0 0 20px rgba(255, 204, 51, 0.35);
            z-index: 100;
        }}
        #star-panel h3 {{
            margin: 0 0 12px 0;
            color: #ffcc33;
        }}
    &lt;/style&gt;
&lt;/head&gt;
&lt;body&gt;
    &lt;div id="container"&gt;&lt;/div&gt;
    &lt;div id="info-panel"&gt;
        &lt;button id="close-info"&gt;&amp;times;&lt;/button&gt;
        &lt;div id="info-content"&gt;&lt;/div&gt;
    &lt;/div&gt;
    &lt;div id="star-panel"&gt;
        &lt;h3&gt;🌟 Host Star Info&lt;/h3&gt;
        &lt;div id="star-content"&gt;&lt;/div&gt;
    &lt;/div&gt;

    &lt;script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"&gt;&lt;/script&gt;
    &lt;script&gt;
        const planetsData = {planets_json};
        const starInfo = {star_json};

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.getElementById('container').appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        camera.position.set(0, 80, 150);
        controls.update();

        const ambientLight = new THREE.AmbientLight(0x404040, 1.0);
        scene.add(ambientLight);
        const sunLight = new THREE.PointLight(0xffffff, 2.5, 2000);
        sunLight.position.set(0, 0, 0);
        scene.add(sunLight);

        // Add Sun
        const sunGeometry = new THREE.SphereGeometry(10, 64, 64);
        const sunMaterial = new THREE.MeshBasicMaterial({{ color: 0xffcc33 }});
        const sun = new THREE.Mesh(sunGeometry, sunMaterial);
        scene.add(sun);

        for (let i = 1; i &lt;= 3; i++) {{
            const glowGeom = new THREE.SphereGeometry(10 + i * 2, 64, 64);
            const glowMat = new THREE.MeshBasicMaterial({{
                color: 0xffaa00,
                transparent: true,
                opacity: 0.4 / i
            }});
            const glowMesh = new THREE.Mesh(glowGeom, glowMat);
            scene.add(glowMesh);
        }}

        // Update Star Panel
        document.getElementById('star-content').innerHTML = Object.keys(starInfo).length ?
            `&lt;p&gt;&lt;b&gt;Name:&lt;/b&gt; ${{starInfo.name || 'Unknown'}}&lt;/p&gt;
             &lt;p&gt;&lt;b&gt;Temperature:&lt;/b&gt; ${{starInfo.temp ? starInfo.temp.toFixed(0) : 'Unknown'}} K&lt;/p&gt;
             &lt;p&gt;&lt;b&gt;Luminosity:&lt;/b&gt; ${{starInfo.luminosity ? starInfo.luminosity.toFixed(2) : 'Unknown'}} L☉&lt;/p&gt;`
            : '&lt;p&gt;Star info not available.&lt;/p&gt;';

        function createTextSprite(text) {{
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 512;
            canvas.height = 128;

            ctx.fillStyle = 'rgba(0, 0, 0, 0)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.font = 'Bold 60px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.shadowColor = 'black';
            ctx.shadowBlur = 10;
            ctx.fillStyle = 'white';
            ctx.fillText(text, 256, 64);

            const texture = new THREE.CanvasTexture(canvas);
            const spriteMaterial = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.scale.set(30, 8, 1);
            return sprite;
        }}

        const planetObjects = [];
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        planetsData.forEach((data, index) =&gt; {{
            const orbitRadius = data.orbit_radius * 20;
            const orbitCurve = new THREE.EllipseCurve(0, 0, orbitRadius, orbitRadius);
            const orbitPoints = orbitCurve.getPoints(128);
            const orbitGeom = new THREE.BufferGeometry().setFromPoints(orbitPoints);
            const orbitColor = data.is_habitable ? 0x00ffff : 0x555555;
            const orbitMat = new THREE.LineBasicMaterial({{
                color: orbitColor,
                transparent: true,
                opacity: 0.6
            }});
            const orbitLine = new THREE.LineLoop(orbitGeom, orbitMat);
            orbitLine.rotation.x = Math.PI / 2;
            scene.add(orbitLine);

            const pSize = Math.max(1.5, data.radius * 1.2);
            const pGeom = new THREE.SphereGeometry(pSize, 32, 32);

            let pColor;
            if (data.habitability_score !== undefined) {{
                if (data.habitability_score &gt; 0.7) pColor = 0x00ff88;
                else if (data.habitability_score &gt; 0.4) pColor = 0x00ffff;
                else pColor = 0x4466aa;
            }} else {{
                pColor = data.is_habitable ? 0x00ffcc : 0x4466aa;
            }}

            const pMat = new THREE.MeshStandardMaterial({{
                color: pColor,
                emissive: pColor,
                emissiveIntensity: 0.3,
                roughness: 0.6,
                metalness: 0.3
            }});
            const pMesh = new THREE.Mesh(pGeom, pMat);
            pMesh.userData = {{ type: 'planet', data: data, index: index, orbitLine: orbitLine }};
            scene.add(pMesh);

            const labelSprite = createTextSprite(data.name);
            scene.add(labelSprite);

            const orbitalSpeed = 0.4 / Math.sqrt(data.orbit_radius);
            const initialAngle = Math.random() * Math.PI * 2;

            planetObjects.push({{
                mesh: pMesh,
                label: labelSprite,
                orbitLine: orbitLine,
                radius: orbitRadius,
                speed: orbitalSpeed,
                angle: initialAngle,
                labelOffset: pSize + 5,
                data: data
            }});
        }});

        // Starfield
        const starFieldGeom = new THREE.BufferGeometry();
        const starFieldMat = new THREE.PointsMaterial({{ color: 0xffffff, size: 0.2, transparent: true, opacity: 0.8 }});
        const starVerts = [];
        for (let i = 0; i &lt; 10000; i++) {{
            starVerts.push((Math.random() - 0.5) * 3000, (Math.random() - 0.5) * 3000, (Math.random() - 0.5) * 3000);
        }}
        starFieldGeom.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
        const starField = new THREE.Points(starFieldGeom, starFieldMat);
        scene.add(starField);

        let selectedPlanet = null;

        window.addEventListener('mousemove', (event) =&gt; {{
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        }});

        window.addEventListener('click', (event) =&gt; {{
            raycaster.setFromCamera(mouse, camera);
            const planetMeshes = planetObjects.map(p =&gt; p.mesh);
            const intersects = raycaster.intersectObjects(planetMeshes);

            if (intersects.length &gt; 0) {{
                selectPlanet(intersects[0].object);
            }}
        }});

        function selectPlanet(planetMesh) {{
            if (selectedPlanet) {{
                selectedPlanet.material.emissiveIntensity = 0.3;
                selectedPlanet.orbitLine.material.opacity = 0.6;
            }}

            selectedPlanet = planetMesh;
            selectedPlanet.material.emissiveIntensity = 1.0;
            selectedPlanet.orbitLine.material.opacity = 1.0;

            const data = selectedPlanet.userData.data;
            const infoPanel = document.getElementById('info-panel');
            const infoContent = document.getElementById('info-content');
            infoContent.innerHTML =
                `&lt;h3&gt;🪐 ${{data.name}}&lt;/h3&gt;
                 &lt;p&gt;&lt;b&gt;Radius:&lt;/b&gt; ${{data.radius.toFixed(2)}} R🜨&lt;/p&gt;
                 &lt;p&gt;&lt;b&gt;Orbit Distance:&lt;/b&gt; ${{data.orbit_radius.toFixed(2)}} AU&lt;/p&gt;
                 &lt;p&gt;&lt;b&gt;Habitable:&lt;/b&gt; ${{data.is_habitable ? '✅ Yes' : '❌ No'}}&lt;/p&gt;
                 ${{data.habitability_score !== undefined ? '&lt;p&gt;&lt;b&gt;Habitability Score:&lt;/b&gt; ' + (data.habitability_score*100).toFixed(0) + '%&lt;/p&gt;' : ''}}
                 ${{data.confidence !== undefined ? '&lt;p&gt;&lt;b&gt;AI Confidence:&lt;/b&gt; ' + (data.confidence*100).toFixed(0) + '%&lt;/p&gt;' : ''}}`;
            infoPanel.style.display = 'block';
        }}

        document.getElementById('close-info').addEventListener('click', () =&gt; {{
            document.getElementById('info-panel').style.display = 'none';
            if (selectedPlanet) {{
                selectedPlanet.material.emissiveIntensity = 0.3;
                selectedPlanet.orbitLine.material.opacity = 0.6;
                selectedPlanet = null;
            }}
        }});

        function animate() {{
            requestAnimationFrame(animate);
            sun.rotation.y += 0.002;

            planetObjects.forEach(obj =&gt; {{
                obj.angle += obj.speed * 0.01;
                const x = Math.cos(obj.angle) * obj.radius;
                const z = Math.sin(obj.angle) * obj.radius;

                obj.mesh.position.set(x, 0, z);
                obj.mesh.rotation.y += 0.015;
                obj.label.position.set(x, obj.labelOffset, z);
            }});

            controls.update();
            renderer.render(scene, camera);
        }}

        window.addEventListener('resize', () =&gt; {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        animate();
    &lt;/script&gt;
&lt;/body&gt;
&lt;/html&gt;
""".strip()
