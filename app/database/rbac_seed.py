"""
RBAC seeding script for StudyGuru Pro
Creates initial modules and sets up basic permissions structure
"""

import asyncio
from sqlalchemy import select
from app.core.database import AsyncSessionLocal, init_db
from app.models.rbac import Module, UserModulePermission
from app.models.user import User, UserAccountType


async def seed_modules():
    """Create initial RBAC modules"""
    async with AsyncSessionLocal() as db:
        try:
            # Define initial modules
            initial_modules = [
                {
                    "name": "User Module",
                    "slug": "user_module",
                    "description": "Module for managing users, permissions, and user-related operations",
                    "is_active": True,
                },
                {
                    "name": "Support Module",
                    "slug": "support_module",
                    "description": "Module for customer support and help desk operations",
                    "is_active": True,
                },
            ]

            for module_data in initial_modules:
                # Check if module already exists
                result = await db.execute(
                    select(Module).where(Module.slug == module_data["slug"])
                )
                existing_module = result.scalar_one_or_none()

                if existing_module:
                    print(f"Module '{module_data['slug']}' already exists, skipping...")
                    continue

                # Create module
                new_module = Module(
                    name=module_data["name"],
                    slug=module_data["slug"],
                    description=module_data["description"],
                    is_active=module_data["is_active"],
                )
                db.add(new_module)
                print(
                    f"✓ Created module: {module_data['name']} ({module_data['slug']})"
                )

            await db.commit()
            print("✅ Modules seeded successfully!")

        except Exception as e:
            await db.rollback()
            print(f"❌ Error seeding modules: {e}")
            raise


async def assign_executive_permissions():
    """Assign full permissions to all EXECUTIVE users for all modules"""
    async with AsyncSessionLocal() as db:
        try:
            # Get all EXECUTIVE users
            result = await db.execute(
                select(User).where(User.account_type == UserAccountType.EXICUTIVE)
            )
            executive_users = result.scalars().all()

            if not executive_users:
                print("No EXECUTIVE users found. Skipping permission assignment...")
                return

            # Get all active modules
            modules_result = await db.execute(
                select(Module).where(Module.is_active == True)
            )
            modules = modules_result.scalars().all()

            if not modules:
                print("No modules found. Please run seed_modules() first.")
                return

            assigned_count = 0
            for user in executive_users:
                for module in modules:
                    # Check if permission already exists
                    perm_result = await db.execute(
                        select(UserModulePermission).where(
                            UserModulePermission.user_id == user.id,
                            UserModulePermission.module_id == module.id,
                        )
                    )
                    existing_perm = perm_result.scalar_one_or_none()

                    if existing_perm:
                        continue

                    # Create full permissions for EXECUTIVE
                    permission = UserModulePermission(
                        user_id=user.id,
                        module_id=module.id,
                        can_read=True,
                        can_insert=True,
                        can_update=True,
                        can_delete=True,
                    )
                    db.add(permission)
                    assigned_count += 1
                    print(
                        f"✓ Assigned full permissions to {user.email} for {module.name}"
                    )

            await db.commit()
            print(f"✅ Assigned {assigned_count} permission(s) to EXECUTIVE users!")

        except Exception as e:
            await db.rollback()
            print(f"❌ Error assigning permissions: {e}")
            raise


async def seed_rbac():
    """Seed RBAC system (modules and permissions)"""
    print("\n🌱 Starting RBAC seed process...")

    # Seed modules
    print("📦 Seeding modules...")
    await seed_modules()

    # Assign permissions to EXECUTIVE users
    print("\n👑 Assigning permissions to EXECUTIVE users...")
    await assign_executive_permissions()

    print("✅ RBAC seed process completed!")


async def main():
    """Run RBAC seeding functions"""
    print("🌱 Starting RBAC seed process...\n")

    # Initialize database connection
    await init_db()

    # Seed RBAC system
    await seed_rbac()

    print("\n🎉 RBAC seed process completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
