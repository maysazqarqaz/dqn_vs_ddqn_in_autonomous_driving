import carla
import random
import numpy as np
from preprocess import FrameStack

class CarlaEnv:
    SHOW_CAM = True
    def __init__(self, port=2000, tm_port=8000):
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(60.0)
        self._tm_port = tm_port

        if self.client.get_world().get_map().name != "Carla/Maps/Town05":
            self.world = self.client.load_world("Town05")
        else:
            self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.front_camera = None
        self.collision_hist = []

        self.npc_vehicles = []
        self.npc_walkers = []       
        self.npc_walker_ids = []

        self.frame_stack = FrameStack(stack_size=8)
        self.action_space = 6

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    # NPC traffic 
    def _spawn_npc(self):
        traffic_manager = self.client.get_trafficmanager(self._tm_port)
        traffic_manager.set_synchronous_mode(True)
        
        vehicle_bps = self.blueprint_library.filter("vehicle.*")
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for sp in spawn_points:
            if len(self.npc_vehicles) >= 50:
                break
            bp = random.choice(vehicle_bps)
            npc = self.world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True, traffic_manager.get_port())
                self.npc_vehicles.append(npc)

        self.world.tick()

        walker_bps = self.blueprint_library.filter("walker.pedestrian.*")
        walker_controller_bp = self.blueprint_library.find("controller.ai.walker")
        walker_spawn_transforms = []
        for _ in range(60):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                walker_spawn_transforms.append(carla.Transform(loc))

        spawned_walkers = []
        for transform in walker_spawn_transforms:
            if len(spawned_walkers) >= 30:
                break
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(bp, transform)
            if walker:
                spawned_walkers.append(walker)

        self.world.tick()

        for walker in spawned_walkers:
            controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
            self.npc_walkers.append((walker, controller))

        self.world.tick()

        for _, controller in self.npc_walkers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.4)

        self.world.tick()

        print(f"Spawned {len(self.npc_vehicles)} NPC vehicles, {len(self.npc_walkers)} pedestrians")

    # Sensor callbacks
    def process_img(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        self.front_camera = array[:, :, :3].copy()

    def collision_data(self, event):
        print(f"[COLLISION] {event.other_actor.type_id}")
        self.collision_hist.append(event)

    # Episode management
    def reset(self):
        self.collision_hist = []
        self.front_camera = None

        # Destroy only the ego (agent) vehicle actors — NPCs persist
        self._destroy_ego()

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicle_bp = self.blueprint_library.filter("model3")[0]

        self.vehicle = None
        for sp in spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
            if self.vehicle is not None:
                break

        if self.vehicle is None:
            raise RuntimeError("No free spawn point found for ego vehicle")

        self.vehicle.set_autopilot(False, self._tm_port)

    # Spawn NPCs after ego vehicle is already placed
        if len(self.npc_vehicles) == 0:
            self._spawn_npc()

        # Camera
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(lambda data: self.process_img(data))

        # Collision sensor
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        for _ in range(20):
            self.world.tick()
            if self.front_camera is not None:
                break
        else:
            raise RuntimeError("Camera never delivered a frame after 20 ticks")

        self.episode_step = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.frame_stack.reset(self.front_camera)

    def apply_action(self, action):
        throttle = 0.5
        steer = 0.0
        brake = 0.0

        if action == 0:        # Steer left
            steer = -1.0
        elif action == 1:      # Go straight
            steer = 0.0
        elif action == 2:      # Steer right
            steer = 1.0
        elif action == 3:      # Slow down and steer left
            throttle = 0.2
            steer = -1.0
        elif action == 4:      # Slow down and go straight
            throttle = 0.2
        elif action == 5:      # Slow down and steer right
            throttle = 0.2
            steer = 1.0

        self.vehicle.apply_control( # type: ignore
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

    def step(self, action):
        self.apply_action(action)
        self.world.tick()
        self.episode_step += 1

        next_state = self.frame_stack.step(self.front_camera)

        if self.episode_step >= 600:
            return next_state, 250, True, {}

        # Collision
        if len(self.collision_hist) != 0:
            return next_state, -20, True, {}

        return next_state, 5, False, {}

    # Cleanup
    def _destroy_ego(self):
        # Stop sensors before destroying 
        for sensor in [self.camera, self.collision_sensor]:
            if sensor is not None:
                try:
                    stop_fn = getattr(sensor, "stop", None)
                    if stop_fn is not None:
                        stop_fn()
                    sensor.destroy()
                except Exception:
                    pass
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
        self.camera = None
        self.collision_sensor = None
        self.vehicle = None
        self.world.tick()

    def destroy(self):
        self._destroy_ego()

        for actor in self.npc_vehicles:
            try:
                actor.destroy()
            except Exception:
                pass
        self.npc_vehicles = []

        for walker, controller in self.npc_walkers:
            try:
                controller.stop()
                controller.destroy()
            except Exception:
                pass
            try:
                walker.destroy()
            except Exception:
                pass
        self.npc_walkers = []

