/**
 * Weapon Equip System for Three.js
 *
 * Equips weapons with embedded Grip nodes to character hand sockets.
 * Core equation: ToolWorld = HandSocketWorld · inverse(GripLocal)
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export class WeaponEquipSystem {
    constructor() {
        this.loader = new GLTFLoader();
        this.weaponCache = new Map(); // Cache loaded weapons
        this.equippedWeapons = new Map(); // Currently equipped: socket -> weapon
    }

    /**
     * Load a weapon GLB (with caching)
     * @param {string} url - URL to weapon GLB
     * @returns {Promise<THREE.Object3D>} - Cloned weapon scene
     */
    async loadWeapon(url) {
        // Check cache
        if (this.weaponCache.has(url)) {
            return this.weaponCache.get(url).clone();
        }

        // Load
        const gltf = await new Promise((resolve, reject) => {
            this.loader.load(url, resolve, undefined, reject);
        });

        // Cache the original
        this.weaponCache.set(url, gltf.scene);

        // Return clone
        return gltf.scene.clone();
    }

    /**
     * Find the Grip node in a weapon scene
     * @param {THREE.Object3D} weapon - Weapon scene root
     * @returns {THREE.Object3D|null} - Grip node or null
     */
    findGripNode(weapon) {
        let grip = null;
        weapon.traverse((child) => {
            if (child.name === 'Grip' || child.name === 'grip') {
                grip = child;
            }
        });
        return grip;
    }

    /**
     * Get grip extras metadata
     * @param {THREE.Object3D} weapon - Weapon scene root
     * @returns {Object} - Extras data
     */
    getGripExtras(weapon) {
        // Check root userData
        if (weapon.userData && weapon.userData.grip_extras) {
            try {
                return JSON.parse(weapon.userData.grip_extras);
            } catch (e) {
                return weapon.userData.grip_extras;
            }
        }

        // Check Grip node
        const grip = this.findGripNode(weapon);
        if (grip && grip.userData) {
            return grip.userData;
        }

        return {};
    }

    /**
     * Equip weapon to a hand socket
     *
     * @param {THREE.Object3D} weapon - Weapon scene (must have Grip node)
     * @param {THREE.Object3D} socket - Hand socket (Object3D parented to hand bone)
     * @param {Object} options - Options
     * @param {boolean} options.keepScale - Preserve weapon scale (default: true)
     */
    equip(weapon, socket, options = {}) {
        const { keepScale = true } = options;

        // Find Grip node
        const grip = this.findGripNode(weapon);
        if (!grip) {
            console.warn('Weapon has no Grip node, attaching at origin');
            socket.add(weapon);
            return;
        }

        // Ensure matrices are up to date
        weapon.updateMatrixWorld(true);
        socket.updateMatrixWorld(true);

        // Get Grip local transform (relative to weapon root)
        const gripLocal = new THREE.Matrix4();
        gripLocal.copy(grip.matrix);

        // Compute inverse of grip local
        const invGripLocal = new THREE.Matrix4();
        invGripLocal.copy(gripLocal).invert();

        // Compute weapon world transform: ToolWorld = SocketWorld * inverse(GripLocal)
        const socketWorld = new THREE.Matrix4();
        socketWorld.copy(socket.matrixWorld);

        const weaponWorld = new THREE.Matrix4();
        weaponWorld.multiplyMatrices(socketWorld, invGripLocal);

        // Apply to weapon
        weapon.matrixAutoUpdate = false;
        weapon.matrix.copy(weaponWorld);
        weapon.matrixWorld.copy(weaponWorld);

        // Parent to socket (this will convert to local space)
        socket.add(weapon);

        // Convert world matrix to local (relative to socket)
        const socketWorldInverse = new THREE.Matrix4();
        socketWorldInverse.copy(socket.matrixWorld).invert();
        weapon.matrix.multiplyMatrices(socketWorldInverse, weaponWorld);

        // Decompose for inspection
        const pos = new THREE.Vector3();
        const quat = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        weapon.matrix.decompose(pos, quat, scale);

        weapon.position.copy(pos);
        weapon.quaternion.copy(quat);
        if (keepScale) {
            weapon.scale.set(1, 1, 1);
        } else {
            weapon.scale.copy(scale);
        }

        weapon.matrixAutoUpdate = true;

        // Track equipped weapon
        this.equippedWeapons.set(socket.uuid, weapon);

        console.log('Equipped weapon:', {
            weaponName: weapon.name,
            socketName: socket.name,
            localPos: [pos.x.toFixed(3), pos.y.toFixed(3), pos.z.toFixed(3)],
        });
    }

    /**
     * Unequip weapon from socket
     * @param {THREE.Object3D} socket - Hand socket
     */
    unequip(socket) {
        const weapon = this.equippedWeapons.get(socket.uuid);
        if (weapon) {
            socket.remove(weapon);
            this.equippedWeapons.delete(socket.uuid);

            // Optionally dispose
            weapon.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => m.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            });
        }
    }

    /**
     * Hot-swap: unequip current and equip new
     * @param {string} weaponUrl - URL to new weapon GLB
     * @param {THREE.Object3D} socket - Hand socket
     */
    async swap(weaponUrl, socket) {
        this.unequip(socket);
        const weapon = await this.loadWeapon(weaponUrl);
        this.equip(weapon, socket);
        return weapon;
    }

    /**
     * Create a hand socket helper
     * @param {THREE.Bone} handBone - The hand bone
     * @param {Object} offset - Local offset { position, rotation }
     * @returns {THREE.Object3D} - Socket object
     */
    createHandSocket(handBone, offset = {}) {
        const socket = new THREE.Object3D();
        socket.name = 'HandSocket';

        // Apply offset
        if (offset.position) {
            socket.position.copy(offset.position);
        }
        if (offset.rotation) {
            socket.rotation.copy(offset.rotation);
        }

        handBone.add(socket);
        return socket;
    }

    /**
     * Debug: visualize socket and grip transforms
     * @param {THREE.Scene} scene - Scene to add helpers to
     * @param {THREE.Object3D} socket - Hand socket
     * @param {THREE.Object3D} weapon - Equipped weapon (optional)
     */
    addDebugHelpers(scene, socket, weapon = null) {
        // Socket axes
        const socketHelper = new THREE.AxesHelper(0.1);
        socketHelper.name = 'SocketHelper';
        socket.add(socketHelper);

        // Grip axes (if weapon equipped)
        if (weapon) {
            const grip = this.findGripNode(weapon);
            if (grip) {
                const gripHelper = new THREE.AxesHelper(0.08);
                gripHelper.name = 'GripHelper';
                grip.add(gripHelper);
            }
        }
    }
}

/**
 * Usage Example:
 *
 * const equipSystem = new WeaponEquipSystem();
 *
 * // Find hand bone in your character
 * const handBone = character.getObjectByName('mixamorig:RightHand');
 *
 * // Create socket (with optional offset for palm position)
 * const socket = equipSystem.createHandSocket(handBone, {
 *     position: new THREE.Vector3(0, 0.05, 0),
 *     rotation: new THREE.Euler(0, 0, Math.PI / 2)
 * });
 *
 * // Load and equip weapon
 * const sword = await equipSystem.loadWeapon('sword_with_grip.glb');
 * equipSystem.equip(sword, socket);
 *
 * // Hot-swap to different weapon
 * await equipSystem.swap('axe_with_grip.glb', socket);
 *
 * // Unequip
 * equipSystem.unequip(socket);
 */

export default WeaponEquipSystem;
